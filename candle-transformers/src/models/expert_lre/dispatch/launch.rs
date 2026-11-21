//! Launches one MoE layer's routed experts on the device, split into host protocol,
//! launches, and submission.
//!
//! `forward` runs three parts in order. `before` does the host protocol and tells
//! the two host threads what to expect, `record` queues the launches, and `after`
//! submits them. The messages go out before the launches, because neither receiver
//! assumes the summary is already published. Keeping the launches apart from the
//! protocol lets a decode's graph be replayed between an unchanged `before` and
//! `after`. The token-tile width is chosen per launch from the expected rows per
//! expert, since prefill's rows per expert cannot be known without the routing
//! readback this path avoids.

use super::*;

impl Dispatch {
    /// The CPU pool's doorbell and `done` words as the host reads them now:
    /// a doorbell ahead of `done` and of the service's last is a layer the
    /// service never took.
    pub(crate) fn cpu_words(&self) -> Option<(u64, u64)> {
        self.cpu
            .as_ref()
            .map(|c| (c.service.rings().doorbell(), c.service.rings().done()))
    }

    /// The CPU export's block counter as the device holds it now — the ticket
    /// it last counted and how many of that launch's blocks were out — and the
    /// work slot's invocation: what a doorbell that never moved, with `done`
    /// level beside it, is to be told apart by.
    pub(crate) fn cpu_export_words(&self) -> Option<(u32, u32, u32)> {
        let c = self.cpu.as_ref()?;
        let word = c
            .blocks_done
            .stream()
            .memcpy_dtov(&c.blocks_done)
            .ok()?
            .first()
            .copied()?;
        Some((
            (word >> 32) as u32,
            word as u32,
            c.service.rings().header().seq,
        ))
    }

    /// `slot_bytes` is the largest 32-row tile slice of any projection — one
    /// worker's scratch slot. `image_bytes` is each row's slot image size —
    /// what a read-ahead copies; with the promotion ring, read-ahead is on.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &CudaDevice,
        table: Arc<LiveTable>,
        pinned: PinnedRanges,
        slot_bytes: usize,
        image_bytes: &[usize],
        k: usize,
        promo: Option<Arc<PromotionRing>>,
        owner: OwnerTags,
        cpu: Option<CpuPath>,
    ) -> Result<Self> {
        if image_bytes.len() != table.n_rows() {
            candle::bail!(
                "expert dispatch: {} slot image sizes for a live table of {} rows",
                image_bytes.len(),
                table.n_rows()
            );
        }
        let ahead = match &promo {
            Some(_) => Some(AheadBuffers {
                row_layout: device.memcpy_stod(&row_layout(&table, image_bytes))?,
                items: device.memcpy_stod(&vec![0u64; 1 + AHEAD_MAX * AHEAD_ITEM_WORDS])?,
                done: device.memcpy_stod(&vec![0u32; AHEAD_MAX])?,
            }),
            None => None,
        };
        let read_ahead = Arc::new(ReadAheadGate::new(&LinkTime::new(promo.is_none())));
        let n_experts = table.n_experts();
        let ring = Arc::new(SummaryRing::new(n_experts)?);
        let votes = Arc::new(VoteRing::new(n_experts)?);
        let abort = Arc::new(AbortWord::new()?);
        let clock = Arc::new(ReclaimClock::new(StartedRows::mapped(table.n_rows())?));
        // Sized here for the widest wave the engine composes — its prefill
        // ceiling, and as many decode and verify rows again — so the forward
        // never grows it. One workspace serves every layer: each layer's
        // tables are consumed by its own launches before the next layer's
        // bucketize writes them, in stream order.
        let workspace = MoeBucketizeWorkspace::new(device, 2 * MAX_PREFILL_TOKENS, k)?;
        // SAFETY (all three): written by bucketize before any reader.
        let snap = unsafe { device.alloc::<u64>(3 * n_experts)? };
        let remote = unsafe { device.alloc::<i32>(4 * n_experts)? };
        let counters = unsafe { device.alloc::<i32>(3)? };
        let remote_dst = unsafe { device.alloc::<u64>(n_experts)? };
        // SAFETY: each worker writes its slot before reading it.
        let scratch = unsafe { device.alloc::<u8>(WORKERS_PREFILL * slot_bytes)? };
        #[cfg(feature = "profile")]
        let stall = device.memcpy_stod(&vec![0u64; table.n_rows() * STALL_WORDS])?;
        let fault = Arc::new(FaultWord::new()?);
        // The pool waits on a cold expert's publish where a GPU worker would,
        // and gives it up through the same words (§5.4).
        if let Some(c) = &cpu {
            c.service.attach_cold(ColdSource {
                table: table.clone(),
                abort: abort.clone(),
                fault: fault.clone(),
                pinned,
                paged: c.paged.clone(),
                spin_limit: Duration::from_nanos(SPIN_LIMIT_NS),
            });
        }
        Ok(Self {
            table,
            ring,
            votes,
            abort,
            fault,
            clock,
            pass: Arc::new(Mutex::new(PassState {
                pass: 0,
                reserved: 0,
            })),
            served: Arc::new(AtomicU64::new(0)),
            staged: Arc::new(AtomicU64::new(0)),
            pinned,
            workspace: Mutex::new(workspace),
            forward: Mutex::new(ForwardSide {
                seq: 0,
                last_row: None,
                fill: SegmentFill::default(),
            }),
            snap,
            remote,
            counters,
            promo,
            remote_dst,
            scratch,
            slot_bytes,
            owner,
            ahead,
            read_ahead,
            cpu,
            cold_wait: ColdWait::new()?,
            #[cfg(feature = "profile")]
            stall,
            #[cfg(feature = "profile")]
            device: device.clone(),
        })
    }

    /// Read and zero the worker counters. Synchronizes the device — a profile
    /// snapshot, between phases, never inside a forward.
    #[cfg(feature = "profile")]
    pub(crate) fn drain_worker_counters(&self) -> Result<WorkerCounters> {
        let rows = self.device.memcpy_dtov(&self.stall)?;
        let stream = self.device.cuda_stream();
        let (ptr, _g) = self.stall.device_ptr(&stream);
        // SAFETY: the counter block this struct owns; the device is synchronized
        // (the read above), so no worker is adding to it, and the fill is queued
        // on the stream every later worker runs behind.
        unsafe {
            cudarc::driver::result::memset_d8_async(ptr, 0, rows.len() * 8, stream.cu_stream())
        }
        .map_err(candle::Error::wrap)?;
        let mut c = WorkerCounters::default();
        for r in rows.chunks_exact(STALL_WORDS) {
            c.cold_wait_ns += r[0];
            c.items += r[1];
            c.bytes += r[2];
            c.copy_ns += r[3];
            c.launches += r[4];
            c.ahead_items += r[5];
            c.ahead_bytes += r[6];
        }
        Ok(c)
    }

    /// Number this invocation, and start a new pass if its row is not past the
    /// previous one's. Returns `(seq, pass)`.
    pub(super) fn begin_invocation(&self, row: usize) -> Result<(u64, u64)> {
        let (seq, new_pass) = {
            let mut f = self.forward.lock().map_err(|_| {
                candle::Error::Msg("expert dispatch: forward state poisoned".into())
            })?;
            let new_pass = f.last_row.is_none_or(|last| row <= last);
            f.last_row = Some(row);
            let seq = f.seq;
            f.seq += 1;
            (seq, new_pass)
        };
        let mut p = self
            .pass
            .lock()
            .map_err(|_| candle::Error::Msg("expert dispatch: pass state poisoned".into()))?;
        p.reserved += 1;
        if new_pass {
            p.pass += 1;
        }
        Ok((seq, p.pass))
    }

    /// Hold invocation `ticket` until both readers of the summary ring have
    /// read the slot's previous tenant, ticket `ticket - SUMMARY_RING`.
    pub(super) fn hold_for_ring(&self, ticket: u64) -> Result<()> {
        let need = ticket.saturating_sub(SUMMARY_RING as u64);
        // Host time the forward thread spends here is the GPU's, not its own:
        // the readers consume a slot only once the device has written it.
        let t = profile_now();
        let mut backoff = Backoff::new(INTER_LAYER_SPINS);
        loop {
            // Whichever reader is behind: its counter's line is the one to
            // watch.
            let behind = [&self.served, &self.staged]
                .into_iter()
                .map(|c| (c, c.load(Ordering::Acquire)))
                .find(|&(_, at)| at < need);
            let Some((counter, at)) = behind else { break };
            if self.abort.is_raised() {
                candle::bail!("expert pipeline or stager died — the layer cannot be served");
            }
            // SAFETY: the dispatch's own counter, live while it is.
            unsafe { backoff.wait_u64(counter.as_ptr(), at, LOW_POWER_CYCLES) };
        }
        crate::models::profile::pipeline_record("moe:hold_ring", t);
        Ok(())
    }

    /// The routed experts of one MoE layer, on the device.
    ///
    /// `acts` is the layer's q8a128 activation `[n_tokens, hidden]`; `weights`
    /// and `indices` the router's `[n_tokens, k]` output (f32, u32). Returns the
    /// routed sum `[n_tokens, hidden]` at `out_dtype`. The tokens in `decode`
    /// are decode rows, which the residency scoring weights differently.
    ///
    /// Three parts, in this order: [`Self::before`] (the host protocol — number
    /// the invocation, hold its summary-ring slot, enqueue its reclaim ticket,
    /// tell the stager and the pipeline thread), [`Self::record`] (the launches
    /// and nothing else) and [`Self::after`] (flush the submission). The
    /// messages go out before the launches because neither receiver assumes the
    /// summary word is published yet: the pipeline thread waits on the ring slot
    /// and the stager polls it. That split is what lets the launches be recorded
    /// into a graph and replayed between an unchanged `before` and `after`
    /// (`docs/decode_graphs.md` §4.5).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward<'w>(
        &self,
        tx: &mpsc::SyncSender<PipelineMessage>,
        stager: &mpsc::Sender<StagerMsg>,
        acts: Q8a128Operand<'w>,
        weights: &LiveTensor<'_>,
        indices: &LiveTensor<'_>,
        row: usize,
        decode: &DecodeRows,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
        look_ahead: &[LookAhead<'_>],
    ) -> Result<LiveTensor<'w>> {
        let Device::Cuda(cuda_dev) = indices.device() else {
            candle::bail!("expert dispatch: expected a CUDA device")
        };
        let (num_tokens, _) = indices.dims2()?;
        // The hops voted for, `1 ..= voted_hops`, each by exactly one part.
        let voted_hops = look_ahead
            .iter()
            .map(|l| l.first_hop + l.hops - 1)
            .max()
            .unwrap_or(0);
        let covered: usize = look_ahead.iter().map(|l| l.hops).sum();
        if voted_hops > HOPS || covered != voted_hops || look_ahead.iter().any(|l| l.first_hop == 0)
        {
            candle::bail!(
                "expert dispatch: look-ahead parts must cover hops 1..=n once each, n ≤ {HOPS}"
            );
        }
        // The forward thread's host time in each part, on its own span: what an
        // invocation costs the host to issue, apart from the device time
        // `q4e:moe_routed:*` measures.
        let t = profile_now();
        let reserved = self.before(tx, stager, row, num_tokens, voted_hops)?;
        crate::models::profile::pipeline_record("moe:host:before", t);
        let t = profile_now();
        let ys = self.record(
            reserved, acts, weights, indices, decode, out_dtype, wave, look_ahead,
        )?;
        crate::models::profile::pipeline_record("moe:host:record", t);
        let t = profile_now();
        self.after(cuda_dev)?;
        crate::models::profile::pipeline_record("moe:host:after", t);
        Ok(ys)
    }

    /// The host protocol ahead of one invocation's launches: its sequence
    /// number, its summary-ring slot held free of the previous tenant, and the
    /// `Routed` messages to the stager and the pipeline thread. The reclaim
    /// rule's key is not set here: bucketize stores the ticket into its row's
    /// started word when the device begins it (`started.rs`). What is set is
    /// the row's enqueued ticket, which marks it upcoming until the device
    /// begins it — a victim preference, not a key (`ReclaimClock::upcoming`).
    pub(super) fn before(
        &self,
        tx: &mpsc::SyncSender<PipelineMessage>,
        stager: &mpsc::Sender<StagerMsg>,
        row: usize,
        num_tokens: usize,
        voted_hops: usize,
    ) -> Result<Reserved> {
        // A faulted forward not yet failed: nothing is launched on top of it,
        // and refusing before the reservation leaves the protocol untouched.
        self.fault.check()?;
        // The pool parks between steps; this layer's doorbell may be moments
        // away.
        if let Some(c) = &self.cpu {
            c.service.wake();
        }
        let (seq, pass) = self.begin_invocation(row)?;
        let ticket = seq + 1;
        let slot = (seq % SUMMARY_RING as u64) as usize;
        self.hold_for_ring(ticket)?;
        self.clock.enqueued(row, ticket);

        let t = profile_now();
        let word = summary_word(seq);
        stager
            .send(StagerMsg::Routed {
                row,
                slot,
                summary_word: word,
                ticket,
            })
            .map_err(|_| candle::Error::Msg("expert stager died — channel closed".into()))?;
        tx.send(PipelineMessage::Routed(RoutedLayer {
            row,
            pass,
            slot,
            summary_word: word,
            ticket,
            prefill_width: num_tokens > PREFILL_LAUNCH_TOKENS,
            voted_hops,
            submitted_at: profile_now(),
        }))
        .map_err(|_| candle::Error::Msg("expert pipeline thread died — channel closed".into()))?;
        crate::models::profile::pipeline_record("moe:route_handoff", t);
        Ok(Reserved { seq, slot, row })
    }

    /// Submit what [`Self::record`] queued. Both host threads wait on the
    /// summary word bucketize writes, and nothing on the forward thread
    /// synchronizes, so on WDDM nothing else would flush it. With no wave open,
    /// every invocation is submitted. Inside a wave capture this is where a
    /// segment ends, at the invocations the flush schedule names
    /// (`flush_schedule`) — the same ones whether the segment records or runs
    /// eagerly, so a wave is cut after the same layers either way. Between
    /// them a recorded invocation waits for its segment's launch; any other —
    /// an eager segment's, or one issued while the wave is held or paused,
    /// whose launches are already on the stream — is submitted without a cut.
    pub(super) fn after(&self, device: &CudaDevice) -> Result<()> {
        let Some(wave) = device.wave_in_progress() else {
            return device.flush_launches();
        };
        let ends = self
            .forward
            .lock()
            .map_err(|_| candle::Error::Msg("expert dispatch: forward state poisoned".into()))?
            .fill
            .record(wave);
        if ends {
            device.flush_launches()
        } else if device.recording_segment().is_some() {
            Ok(())
        } else {
            device.submit_launches()
        }
    }

    /// The launches of one reserved invocation, and nothing else: bucketize,
    /// gather, the three grouped GEMMs, the SwiGLU and the scatter.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record<'w>(
        &self,
        reserved: Reserved,
        acts: Q8a128Operand<'w>,
        weights: &LiveTensor<'_>,
        indices: &LiveTensor<'_>,
        decode: &DecodeRows,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
        look_ahead: &[LookAhead<'_>],
    ) -> Result<LiveTensor<'w>> {
        let device = indices.device().clone();
        let Device::Cuda(cuda_dev) = &device else {
            candle::bail!("expert dispatch: expected a CUDA device")
        };
        let Reserved { seq, slot, row } = reserved;
        // The invocation's ticket: bucketize's started word, and the CPU
        // class's doorbell and `done` value.
        let ticket = seq + 1;
        let (num_tokens, k) = indices.dims2()?;
        let hidden_dim = acts.cols;
        let table = &*self.table;
        let n_experts = table.n_experts();
        let (gate_dtype, down_dtype) = (table.gate_dtype(row), table.down_dtype(row));
        // **The token-tile width, chosen per launch from the expected rows per
        // expert** — the tile width is the GEMM's weight-reuse factor, so a prefill at
        // ~100–300 rows per expert run at the decode width re-streams and re-dequants
        // every expert 2–4× per projection. Rows per expert are not known without the
        // routing readback this path exists to avoid; `n_tokens·k / E` is what uniform
        // routing would give, a lower bound on the rows of an active expert and, at
        // prefill widths, near it — nearly every expert is active. One width for the
        // three projections, since they share one tile table, and wide only where both
        // dtypes have the wide kernels. Measured at the fixed decode width (RTX 3090):
        // Qwen3-30B-A3B prefill at ×10 ran 11% under wide tiles, Qwen3.8-Flash-Next
        // at ×8–×16 25–33%.
        let n_sub = grouped_int8_n_sub(
            (num_tokens * k) / n_experts.max(1),
            &[gate_dtype, down_dtype],
        );
        let tile_w = 16 * n_sub;
        let compute = cuda_dev.cuda_stream();
        let weights_flat = weights.flatten_all()?.contiguous()?;

        let mut ws = self
            .workspace
            .lock()
            .map_err(|_| candle::Error::Msg("moe bucketize workspace poisoned".into()))?;
        let snap = self.snap.device_ptr(&compute).0;
        let remote = self.remote.device_ptr(&compute).0;
        let counters = self.counters.device_ptr(&compute).0;
        let remote_dst = self.remote_dst.device_ptr(&compute).0;
        // The CPU class for this invocation: the process's split, by value
        // (`cpu_pool::params`). Off — a zero cap — the layer is all-GPU. The
        // cap is a live knob, but the rings were sized at load: past their
        // rows, bucketize would publish results the ring cannot hold.
        let cpu = self.cpu.as_ref().and_then(|c| {
            let (mut split, knobs) = shared()
                .lock()
                .map(|p| (p.split(), p.knobs))
                .unwrap_or((CpuSplit::OFF, Default::default()));
            let tight = knobs.wait_tight_ns;
            split.cap_rows = split.cap_rows.min(c.service.rings().cap_rows as u32);
            if split.cap_rows > 0 && c.explores(knobs.explore_period) {
                split.explore_ns = knobs.explore_ns;
            }
            (split.cap_rows > 0).then(|| {
                c.service
                    .set_sum_by_amax(acts.sum_scale == SumScale::ByAmax);
                (c, split, tight)
            })
        });
        // The item buffers go to bucketize while the predictions are read
        // ahead, and while the CPU pool's experts are promoted behind it —
        // the carry's copies are items too. Without either the walk runs only
        // for a launch with remote experts to claim for, and reads none of
        // the ring's listings.
        let reads_ahead = self.read_ahead.pays();
        let carries = cpu.is_some_and(|(_, split, _)| split.promote != CPU_PROMOTE_NONE);
        let ahead = self
            .ahead
            .as_ref()
            .filter(|_| reads_ahead || carries)
            .map(|a| ReadAhead {
                row_layout: a.row_layout.device_ptr(&compute).0,
                rows: table.n_rows() as i32,
                items: a.items.device_ptr(&compute).0,
                done: a.done.device_ptr(&compute).0,
                predict: reads_ahead,
            });
        // The next rows' votes, ahead of bucketize: its summary word is the
        // store the pipeline thread waits on, so the votes are in its slot by
        // then (`votes`).
        let n = (k + LOOK_AHEAD_MARGIN).min(n_experts);
        let (words, mass) = self.votes.dev_slot(slot);
        for la in look_ahead {
            let at = ((la.first_hop - 1) * n_experts * 4) as u64;
            moe_predict_votes(
                la.rows,
                la.first_col,
                la.hops,
                n_experts,
                k,
                n,
                words + at,
                mass + at,
            )?;
        }
        let g = gpu_span("moe:bucketize", &device);
        moe_bucketize(
            indices,
            n_experts,
            tile_w,
            &mut ws,
            Some(&BucketizeLive {
                gate_row: table.row_ptr(Proj::Gate, row),
                table_plane: table.plane(),
                snap,
                pinned: self.pinned,
                summary: self.ring.dev_slot(slot),
                summary_seq: summary_word(seq),
                remote,
                counters,
                row: row as i32,
                // Every launch is offered the ring; a prompt passing over the
                // table — more claiming experts than the ring's sweep word —
                // claims nothing and stays in scratch, decided by bucketize
                // for this launch alone, so a decode row co-batched with a
                // prompt costs at most a deferred claim on the next narrow
                // launch.
                promo: self.promo.as_ref().map(|p| p.ring()),
                remote_dst,
                started_rows: self.clock.started_ptr(),
                ticket: seq + 1,
                owner: self.owner.check,
                // The ring's read-ahead: what this launch's misses leave of the
                // layer's link window goes to the vetted experts predicted for
                // the rows after the next, copied by the gate launch's workers.
                ahead,
                // The CPU class: bucketize picks the misses the pool takes
                // and publishes them to the work slot, and hands the next
                // launch the ones it promotes behind the pool.
                cpu: cpu.map(|(c, split, _)| BucketizeCpu {
                    split,
                    work: c.service.rings().work.dev(),
                    dst_row: c.dst_row.device_ptr(&compute).0,
                    carry: c.carry.device_ptr(&compute).0,
                    tok_first: c.tok_first.device_ptr(&compute).0,
                    in_tok: c.in_tok.device_ptr(&compute).0,
                    input_map: c.service.rings().inputs.dev(),
                }),
            }),
            decode,
        )?;
        g.end();

        // ── The expert chain ──
        let a_ub = num_tokens * k;
        // Tight data-independent tile bound: full tiles ≤ ⌈a_ub/tile_w⌉ and each
        // expert adds at most one partial tile.
        let launch_tiles = a_ub.min(a_ub.div_ceil(tile_w) + n_experts);
        let hp = ws.header.device_ptr(&compute).0;
        let live = |proj: Proj| MoeLive {
            abort: self.abort.ptr(),
            fault: self.fault.ptr(),
            live_row: table.row_ptr(proj, row),
            remote,
            remote_dst,
            header: hp,
            counter: counters + 4 * proj as u64,
            scratch: self.scratch.device_ptr(&compute).0,
            slot_bytes: self.slot_bytes as u64,
            dst_offset: table.offset(proj, row),
            #[cfg(feature = "profile")]
            stall: self.stall.device_ptr(&compute).0 + (row * STALL_WORDS * 8) as u64,
            #[cfg(not(feature = "profile"))]
            stall: 0,
            // The gate launch is the layer's first over the remote experts —
            // its workers wait on the cold ones while the link is free — so it
            // carries the read-ahead.
            ahead: ahead.filter(|_| proj == Proj::Gate).map_or(0, |a| a.items),
            ahead_done: ahead.filter(|_| proj == Proj::Gate).map_or(0, |a| a.done),
            spin_limit_ns: SPIN_LIMIT_NS,
            // The down launch's CPU items pull the pool's results into the
            // rows bucketize left the CPU experts.
            cpu_done: cpu
                .filter(|_| proj == Proj::Down)
                .map_or(0, |(c, _, _)| c.service.rings().done.dev()),
            cpu_seq: ticket,
            cpu_rows: cpu.map_or(0, |(c, _, _)| c.service.rings().results.dev()),
            cpu_dst_row: cpu.map_or(0, |(c, _, _)| c.dst_row.device_ptr(&compute).0),
            cpu_alive: cpu.map_or(0, |(c, _, _)| c.service.rings().alive.dev()),
            cpu_tight_ns: cpu.map_or(0, |(_, _, t)| t),
            cpu_wait: cpu.map_or(0, |(c, _, _)| c.service.rings().wait.dev()),
            // The gate launch is where a cold expert's wait happens; the up
            // and down launches find it published.
            cold_wait: if proj == Proj::Gate {
                self.cold_wait.dev()
            } else {
                0
            },
            workers: workers_for(num_tokens) as i32,
            row: row as i32,
        };

        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::checkpoint_q8a128;
            let (r, c, n) = (acts.rows, acts.cols, acts.byte_len());
            let nm = candle::tensor_assert::site("moe.norm_out.L", row);
            acts.with_device_ptr(cuda_dev, |p| unsafe {
                checkpoint_q8a128(nm, p, r, c, n, cuda_dev)
            })?;
        }
        // The CPU experts' inputs out to the pool's ring — each distinct token
        // bucketize chose, once, straight from the MoE input — then its
        // doorbell, ahead of the gather: the pool computes while the GPU runs
        // the layer's own experts. A layer with no CPU expert exports, fences
        // and rings nothing.
        if let Some((c, _, _)) = cpu {
            let g = gpu_span("moe:cpu_export", &device);
            let rings = c.service.rings();
            let tiles = hidden_dim / 128;
            let rows = num_tokens.min(rings.cap_rows);
            let hp = ws.header.device_ptr(&compute).0;
            let ip = c.in_tok.device_ptr(&compute).0;
            let bp = c.blocks_done.device_ptr(&compute).0;
            let status = acts.with_device_ptr(cuda_dev, |src| {
                // SAFETY: every address is a live allocation on this device or
                // mapped for it, sized for `rows` rows; the stream is the
                // forward's.
                Ok(unsafe {
                    run_moe_cpu_export(
                        src as *const c_void,
                        tiles as i32,
                        ip as *const c_void,
                        hp as *const c_void,
                        rings.rows.dev() as *mut c_void,
                        rings.cap_rows as u32,
                        rings.doorbell.dev() as *mut c_void,
                        ticket,
                        bp as *mut c_void,
                        (rows * tiles)
                            .div_ceil(EXPORT_WARPS)
                            .clamp(1, EXPORT_MAX_BLOCKS) as i32,
                        compute.cu_stream() as *mut c_void,
                    )
                })
            })?;
            if status != 0 {
                candle::bail!("expert dispatch: the CPU export launch failed ({status})");
            }
            g.end();
        }
        let g = gpu_span("moe:gather", &device);
        let stacked = fused_moe_gather_q8a128(&acts, &ws.tok_ids, a_ub, cuda_dev, wave_root(wave))?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::checkpoint_q8a128;
            let (r, c, n) = (stacked.rows, stacked.cols, stacked.byte_len());
            let nm = candle::tensor_assert::site("moe.gathered.L", row);
            stacked.with_device_ptr(cuda_dev, |p| unsafe {
                checkpoint_q8a128(nm, p, r, c, n, cuda_dev)
            })?;
        }

        let g = gpu_span("moe:gate", &device);
        let gate_out = grouped_qmatmul_dev_q8a128(
            &stacked,
            &self.snap,
            0,
            n_experts,
            gate_dtype,
            table.gate_nrows(),
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            launch_tiles,
            n_sub,
            Some(&live(Proj::Gate)),
            cuda_dev,
        )?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::{capture_gate_gemm, GemmCall};
            capture_gate_gemm(
                &GemmCall {
                    layer: row,
                    stacked: &stacked,
                    weight_ptrs: &self.snap,
                    expert_base: 0,
                    num_experts: n_experts,
                    weight_dtype: gate_dtype,
                    weight_nrows: table.gate_nrows(),
                    tile_expert: &ws.tile_expert,
                    tile_b_start: &ws.tile_b_start,
                    tile_b_cnt: &ws.tile_b_cnt,
                    launch_tiles,
                    out: &gate_out,
                },
                cuda_dev,
            )?;
        }
        let g = gpu_span("moe:up", &device);
        let up_out = grouped_qmatmul_dev_q8a128(
            &stacked,
            &self.snap,
            n_experts,
            n_experts,
            gate_dtype, // up shares gate's KO dtype
            table.gate_nrows(),
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            launch_tiles,
            n_sub,
            Some(&live(Proj::Up)),
            cuda_dev,
        )?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::checkpoint;
            use candle::tensor_assert::site;
            checkpoint(
                site("moe.up_out.L", row),
                &up_out,
                &[("gate_out", &gate_out)],
                cuda_dev,
            )?;
        }
        let g = gpu_span("moe:silu", &device);
        // Raw Σx — a language model's SwiGLU intermediate stays orders of
        // magnitude below f16's 65504; the down matmul reads this operand's own
        // `sum_scale`, so the two agree by construction.
        let inter_acts = silu_mul_q8a128(
            &gate_out,
            &up_out,
            cuda_dev,
            gate_out.cuda_backing(),
            SumScale::Raw,
        )?;
        g.end();
        let g = gpu_span("moe:down", &device);
        let down_out = grouped_qmatmul_dev_q8a128(
            &inter_acts,
            &self.snap,
            2 * n_experts,
            n_experts,
            down_dtype,
            table.down_nrows(),
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            launch_tiles,
            n_sub,
            Some(&live(Proj::Down)),
            cuda_dev,
        )?;
        g.end();
        // The CPU items write the pool's F32 rows straight into this output;
        // an output of any other type would leave those rows unwritten.
        if cpu.is_some() {
            expect_dtype(&down_out, DType::F32, "moe down (CPU result rows)")?;
        }
        #[cfg(feature = "tensor-assert")]
        crate::models::nan_capture::checkpoint(
            candle::tensor_assert::site("moe.down_out.L", row),
            &down_out,
            &[],
            cuda_dev,
        )?;
        // The int8 matmul emits F32 and the scatter reads F32, narrowing once
        // at its store into `ys`'s dtype (hot-path invariant 1). The combine
        // target is defined in full by the scatter, so it is uninitialised
        // (invariant 6).
        let ys = wave_empty((num_tokens, hidden_dim), out_dtype, &device, wave)?;
        let g = gpu_span("moe:scatter", &device);
        fused_deterministic_scatter(
            &ys,
            &down_out,
            &ws.perm,
            &weights_flat,
            &ws.rw_ids,
            &ws.token_starts,
            num_tokens,
            cuda_dev,
        )?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        crate::models::nan_capture::checkpoint(
            candle::tensor_assert::site("moe.routed_out.L", row),
            &ys,
            &[("down_out", &down_out)],
            cuda_dev,
        )?;
        Ok(ys)
    }
}
