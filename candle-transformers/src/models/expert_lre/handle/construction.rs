//! Builds the expert cache: fills its tiers from the model pack and checks every
//! precondition before the cache serves a token.
//!
//! The cache refuses any expert format its kernels cannot run, and refuses a
//! pack packed for another numeric mode or expert count, so a load fails with its
//! reason instead of the first MoE forward failing with a symptom. The pinned set
//! must be affordable before anything loads, because pinned layers have no warm
//! copy and are never evicted, so a zone too small for them stops every later
//! load. The warm tier is sized to what the machine can spare and to the evictable
//! experts, since a miss is the only thing it serves, and a cache that holds every
//! expert wants no warm tier at all. The live table is built for every cache,
//! because one expert path serves them all.

use super::*;

impl ExpertCache {
    /// A build without `cuda` has no expert path: the experts run only as the
    /// device-side grouped GEMM over the live pointer table.
    #[cfg(not(feature = "cuda"))]
    pub fn new(_setup: ExpertCacheSetup<'_>) -> Result<Self> {
        candle::bail!(
            "MoE expert cache: the routed experts run only on the device — build with the \
             `cuda` feature"
        )
    }

    /// Create a new expert cache with a background pipeline thread.
    ///
    /// Fills the warm and hot tiers from the model pack's expert section and
    /// builds the live pointer table over the hot tier. Requires an actual CUDA
    /// device: a cuda-feature build handed a CPU device fails here rather than
    /// later.
    #[cfg(feature = "cuda")]
    pub fn new(setup: ExpertCacheSetup<'_>) -> Result<Self> {
        let ExpertCacheSetup {
            pack,
            zone,
            device,
            experts_per_layer,
            experts_used,
            progress,
            int8mode,
            mapped_bytes,
            offloaded_bytes,
        } = setup;
        // Experts run only on the KO int8 tensor-core path: the FP GEMX kernel
        // was deleted with the float fast path, so an `Off` slot would repack
        // to a layout no kernel can run and every slot construction downstream
        // would fail one expert at a time (`from_qtensor_repacked: only KO
        // twins are runnable`). Refuse here, at the one place every routed
        // model passes through, so the load fails with the reason instead of
        // the first MoE forward failing with the symptom. `Off` remains valid
        // for dense projections, which never build this cache.
        if int8mode == Int8Mode::Off {
            candle::bail!(
                "expert cache: Int8Mode::Off has no expert kernel — the FP GEMX \
                 expert path was removed, so routed (MoE) models require an int8 \
                 mode (Precision or Performance). This device/model combination \
                 selected Off; pass an explicit int8 mode that this GPU supports."
            );
        }
        // The section was packed for one numeric mode and one expert count; a
        // load asking for another would read records laid out for something
        // else.
        let header = pack.header();
        if header.int8_mode != int8mode as u32
            || header.experts_per_layer as usize != experts_per_layer
        {
            candle::bail!(
                "expert cache: the pack holds {} experts per layer packed for int8 mode {}, the \
                 load asks for {experts_per_layer} at {int8mode:?}",
                header.experts_per_layer,
                header.int8_mode
            );
        }
        let num_moe_layers = header.layers.len();
        let num_slots = zone.capacity();
        // **The pinned set must be affordable before anything is loaded.**
        //
        // `PINNED_LAYERS` is fixed, and those layers have no slot in the warm
        // tier and are never evicted — so a zone too small to hold them plus
        // one layer's worst-case routed set does not degrade, it stops: every
        // resident slot ends up holding an expert the eviction scan is forbidden
        // to touch, and every load from then on fails permanently. The zone's
        // floor states the requirement but `WeightZone::new` does not raise a
        // smaller opening to meet it, and only one of the three MoE loaders
        // clamps its own capacity — so the check belongs here, on the path all
        // of them take.
        //
        // **CUDA only**, because a zone of zero slots is what a CPU device
        // always produces — there is no weight zone off the GPU. Checking it
        // first turned the deliberate "this build has CUDA compiled in but was
        // given a CPU device" message below into "affords 0 expert slots, below
        // the floor of 385", which names the symptom and hides the cause.
        let floor =
            minimum_resident_slots(experts_per_layer).min(num_moe_layers * experts_per_layer);
        if matches!(device, Device::Cuda(_)) && num_slots < floor {
            candle::bail!(
                "MoE expert cache: this device affords {num_slots} expert slots, below the \
                 floor of {floor} — {} permanently resident layers of {experts_per_layer} \
                 experts plus one layer's worst-case routed set. Below it the eviction scan \
                 has no candidates and every load fails.",
                pinned_layer_count(num_moe_layers),
            );
        }
        let mut inner = ExpertCacheInner::new(zone, num_moe_layers, experts_per_layer);

        // ── The copy stream: every promotion and relocation, in order ──
        let Device::Cuda(cuda_dev_ref) = device else {
            candle::bail!(
                "MoE expert cache: this build has CUDA compiled in but was given {device:?}. \
                 The expert tiers are all device-side or DMA-bound, so there is no CPU path."
            )
        };
        let copy_stream = cuda_dev_ref
            .cuda_context()
            .new_stream()
            .map_err(candle::Error::wrap)?;

        // ── CUDA startup: the pack, then the resident tiers from it ──
        let (pack, warm, pad, residency, layer_geometries, all_resident, can_fault, order) =
            if let Device::Cuda(cuda_dev) = device {
                let geoms = geometries_of(pack.header())?;
                let total_experts = num_moe_layers * experts_per_layer;
                let all_resident = num_slots >= total_experts;

                // The mapping is the model pack's GGUF part — every tensor but
                // the experts — so all of it is live weight except what a
                // bounded cache of the model's own serves (`offloaded_bytes`):
                // those pages are not the page cache's to keep. That figure is
                // subtracted from the host-RAM budget the warm tier is then
                // sized out of, so overstating it would take the RAM away from
                // the tier whose whole job is to keep experts off the disk.
                let live_weight_bytes = mapped_bytes.saturating_sub(offloaded_bytes);
                candle::vram::set_weights_mmap(live_weight_bytes);
                tracing::info!(
                    target: "candle_transformers::expert_lre",
                    mapped_gib = mapped_bytes as f64 / 1e9,
                    offloaded_gib = offloaded_bytes as f64 / 1e9,
                    live_gib = live_weight_bytes as f64 / 1e9,
                    "expert cache: live weight bytes of the mapping"
                );

                // The pack's record layout **is** the VRAM slot's layout: same
                // three projections, same aligned offsets. One geometry, so a
                // load is a read and a copy with nothing rearranged in between.
                let slot_bytes = slot_bytes_for(&geoms);
                let layouts: Vec<RecordLayout> =
                    (0..num_moe_layers).map(|l| pack.layout(l)).collect();

                // The warm tier is sized by what the machine can spare, not by
                // what residency demands — the cold tier serves every expert at
                // any warm size, including zero.
                let stride = candle::direct_io::round_up_sector(slot_bytes);
                // **The pad before the warm tier, not after.** It is mandatory —
                // it is what makes a cold expert computable — and
                // [`PAD_LAYERS`] layers of slots; the warm tier is elastic and
                // pins up to the driver's page-lock ceiling, so anything pinned
                // after it lands on an exhausted budget. The startup fill
                // borrows the pad as its landing ring before the stager takes it
                // over.
                let pad = Pad::new(
                    if all_resident {
                        STARTUP_RING_SLOTS
                    } else {
                        (PAD_LAYERS * experts_per_layer).max(STARTUP_RING_SLOTS)
                    },
                    stride,
                )?;
                // **A cache that holds every expert in VRAM wants no warm tier
                // at all.** Nothing is ever evicted in that state, so no routed
                // expert is ever read from a host tier. Sizing it anyway would pin the model's size in host RAM
                // to serve nothing, and pay a full-pack read at startup for it.
                // The warm tier's job is covering **misses**, and the pinned
                // prefix never generates one, so it is sized against the
                // evictable set rather than the model. On the 3.6-35B that is
                // 512 fewer experts to aim at — 943 MiB of pinned host memory
                // that used to be spent on experts no load could ever ask for.
                let pinned = pinned_layer_count(num_moe_layers);
                let evictable = total_experts - pinned * experts_per_layer;
                let mut residency =
                    vec![vec![ExpertResidency::default(); experts_per_layer]; num_moe_layers];
                let want_warm = if all_resident {
                    0
                } else {
                    warm_slots_for(stride, evictable)
                };
                // What the startup fill takes into VRAM: the pinned layers,
                // then the rest of the zone spread level across the evictable
                // layers. The warm draw covers the experts it leaves out first.
                // Drawn over every evictable expert: the startup tier is its
                // prefix, and the elastic tier grows into the rest in order.
                let vram = vram_startup_set(
                    num_moe_layers,
                    experts_per_layer,
                    num_slots,
                    pinned,
                    VRAM_DRAW_SEED,
                );
                let order = stratified_membership(
                    num_moe_layers,
                    experts_per_layer,
                    evictable,
                    &vram,
                    pinned,
                    WARM_DRAW_SEED,
                );
                let membership = &order[..want_warm.min(order.len())];
                // Pinned as far as the driver grants and the device can address
                // while keeping both margins, pageable for the rest — see
                // `warm_tier`. Read after the pad, which is charged against it.
                if let Some(report) = progress {
                    report(LoadPhase::PinWarm, 0, 0);
                }
                let mut warm = WarmTier::new(
                    membership.len(),
                    stride,
                    chunk_slots(stride, evictable),
                    gpu_addressable_room(cuda_dev)?,
                );
                // A refusal shortens the draw rather than leaving slots the tier
                // does not have: `ram` must never name a slot outside it.
                let membership = &membership[..membership.len().min(warm.num_slots())];
                let can_fault = can_go_cold(
                    all_resident,
                    membership.len(),
                    evictable,
                    warm.paged_slots(),
                );

                // The eviction policy weighs what a reload would cost, so it has
                // to know which experts the warm tier holds before the first
                // victim is chosen.
                inner.set_warm_backed(membership);
                let mut ring = StartupRing::new(&pad);
                let targets = StartupTargets {
                    inner: &mut inner,
                    warm: &mut warm,
                    residency: &mut residency,
                    membership,
                    vram: &vram,
                    geoms: &geoms,
                    layouts: &layouts,
                    stride,
                };
                // The pack's stride is what the geometry says it is — the
                // buffers above were cut to it.
                if pack.stride() != stride {
                    candle::bail!(
                        "expert cache: the pack's stride is {} B, its geometry makes {stride} B",
                        pack.stride()
                    );
                }
                startup_from_pack(
                    targets,
                    &pack,
                    &mut ring,
                    num_moe_layers,
                    experts_per_layer,
                    cuda_dev,
                    progress,
                )?;
                drop(ring);
                (
                    pack,
                    warm,
                    pad,
                    residency,
                    geoms,
                    all_resident,
                    can_fault,
                    order,
                )
            } else {
                // Refused above, where the copy stream is created; kept as the
                // match's other arm so the binding stays a plain `let`.
                candle::bail!("MoE expert cache: expected a CUDA device, got {device:?}")
            };

        let stats = PipelineStats::new_shared();
        // Seed the resident-expert VRAM gauge with the startup footprint (occupied
        // slots × slot size) so it reads correctly before the pipeline thread's
        // first per-layer refresh. `inner` + `layer_geometries` are still in scope here,
        // before they move into `PipelineState` below.
        #[cfg(feature = "cuda")]
        {
            let occupied = inner.num_slots() - inner.free_len();
            let slot_bytes = layer_geometries
                .iter()
                .map(|g| g.total_repacked_size)
                .max()
                .unwrap_or(0);
            let seeded = occupied * slot_bytes;
            tracing::info!(
                target: "candle_transformers::expert_lre",
                num_slots = inner.num_slots(),
                free_slots = inner.free_len(),
                occupied,
                slot_bytes,
                resident_gib = seeded as f64 / 1e9,
                "expert cache: seeded resident-VRAM gauge"
            );
            // The last of the tail: the cache is built and reporting itself, so
            // the bar lands on its total here rather than at the last expert.
            if let Some(report) = progress {
                let total = num_moe_layers * experts_per_layer;
                report(
                    LoadPhase::FillVram,
                    total + PACK_TAIL_STEPS,
                    total + PACK_TAIL_STEPS,
                );
            }
            if let Ok(mut s) = stats.lock() {
                s.resident_vram_bytes = seeded;
                s.warm_slots = warm.num_slots();
                s.warm_paged_slots = warm.paged_slots();
                s.total_experts = num_moe_layers * experts_per_layer;
                s.moe_layers = num_moe_layers;
                // **The zone's shape, seeded here and not left to the pipeline thread.**
                //
                // These four are what `WeightPlan::from_stats` needs, and it refuses the
                // whole gauge set if any reads zero — correctly, since a zero slot size
                // makes a routed expert look free. The pipeline thread refreshes them
                // per routed layer (`publish_gauges`), but the scheduler reads them
                // before the first forward has routed anything.
                //
                // Left at zero they invert admission. With no weight plan the
                // scheduler's rate planner cannot be armed, and admission falls back to
                // one prefill per pass — so a card *large enough to hold the whole
                // checkpoint* runs waves one row wide. Measured on the 30B-A3B at 72
                // GiB with the gauges unseeded: `decode seqs avg=1.0 max=1` and 32 t/s
                // against a batched ceiling of 518.
                //
                // Every term is known here — the zone is carved before this point and
                // `slot_bytes` is the same figure the resident gauge above is a multiple
                // of.
                let zone_slot_bytes = inner.zone.slot_bytes();
                s.expert_slot_bytes = zone_slot_bytes;
                s.zone_bytes = inner.zone.capacity() * zone_slot_bytes;
                s.zone_min_bytes = inner.zone.min_capacity() * zone_slot_bytes;
                s.zone_max_bytes = inner.zone.limit() * zone_slot_bytes;
            }
        }

        // ── The live table, over where the fill just put every expert ──
        //
        // Built for every cache, all-resident or not: there is one expert path,
        // and the table is how the device finds a weight. Every condition that
        // path depends on is checked here and refused with its reason.
        let table = Arc::new(LiveTable::new(
            &layer_geometries,
            experts_per_layer,
            experts_used,
        )?);
        let mut places = Residency::new(table.clone());
        // Shared from here: the pageable copies' addresses go to the CPU pool,
        // and the tier is held by what reads them.
        let warm = Arc::new(warm);
        // The warm tier's pinned block, as the fill left it, becomes the pad's
        // resident region: one pinned tier the stager labels — residents fixed
        // through a wave, churn for its cold reads — re-chosen between waves
        // (`resident`). The pageable rest stays the warm tier's.
        let mut pad = pad;
        let churn = pad.churn_slots();
        pad.attach_resident(warm.clone());
        let mut residents = Vec::new();
        let paged = Arc::new(PagedImages::new(
            num_moe_layers,
            experts_per_layer,
            warm.clone(),
        ));
        let mut paged_members = Vec::new();
        for (row, layer) in residency.iter().enumerate() {
            for (expert, res) in layer.iter().enumerate() {
                if let Some(slot) = res.vram {
                    places.set_vram(row, expert, Some((slot, inner.slot_base(slot))));
                }
                if let Some(ws) = res.ram {
                    if warm.is_pinned(ws) {
                        let slot = churn + ws;
                        places.set_pad(row, expert, Some((slot, pad.slot_addr(slot))));
                        residents.push((slot, row, expert));
                    } else {
                        places.set_warm(row, expert, Some(ws));
                        paged.set(row, expert, warm.slot_ref(ws, 1).as_ptr() as u64);
                        paged_members.push((ws, row, expert));
                    }
                }
            }
        }
        places.publish_all();
        let places = Arc::new(Mutex::new(places));
        // Four layers of ring indices: up to two layers' worth stocked with
        // free slots (a cold layer, with the next one's refill still behind
        // the GPU), the rest room for the slots the GPU has taken and this
        // thread has not yet collected. A cache holding every expert misses
        // nothing.
        let promo = if all_resident {
            None
        } else {
            Some(Arc::new(PromotionRing::new(
                4 * experts_per_layer,
                num_moe_layers,
                experts_per_layer,
            )?))
        };
        // A lazy victim is named to the device by `row · experts_per_layer +
        // expert` and decoded there against the table's width: the two must be
        // one number, or a claim retargets another expert's entries.
        if table.n_experts() != experts_per_layer {
            candle::bail!(
                "MoE expert cache: the live table is {} experts wide but the cache has {} per \
                 layer — the promotion ring's victim indices would name the wrong experts",
                table.n_experts(),
                experts_per_layer
            );
        }
        // The slot tags bucketize's owner check reads: every tenant the
        // startup fill installed, then every one a slot gains from here on.
        #[cfg(feature = "tensor-assert")]
        let owner = {
            let owners = Arc::new(SlotOwners::new(inner.zone.limit())?);
            inner.attach_owners(owners.clone());
            let slot_bytes = inner.zone.slot_bytes();
            OwnerTags::new(owners, inner.slot_base(0) + slot_bytes as u64, slot_bytes)
        };
        #[cfg(not(feature = "tensor-assert"))]
        let owner = OwnerTags::none();
        let image_bytes: Vec<usize> = layer_geometries.iter().map(|g| slot_offsets(g).3).collect();
        // The link's rate: what read-ahead's window is sized from, and the CPU
        // class's copy cost — measured only where either runs.
        let cpu_enabled = cpu_params().lock().map(|p| p.enabled).unwrap_or(false);
        let link_rate = if promo.is_some() || cpu_enabled {
            measure_link_rate(cuda_dev_ref, &copy_stream)?
        } else {
            0.0
        };
        // The CPU class (`docs/cpu_expert_compute.md`): the host pool takes the
        // cheap pinned misses bucketize gives it, when the process's split is on.
        let cpu = match start_cpu_service(&layer_geometries, link_rate)? {
            Some(service) => Some(CpuPath::new(
                cuda_dev_ref,
                service,
                paged.clone(),
                experts_used,
            )?),
            None => None,
        };
        // The down launch's CPU items wait on the pool, and give its results up
        // through the same fault word, whatever is resident.
        let can_fault = can_fault || cpu.is_some();
        let dispatch = Dispatch::new(
            cuda_dev_ref,
            table,
            [warm.pinned_range(), pad.range()],
            row_tile_bytes_for(&layer_geometries),
            &image_bytes,
            experts_used,
            promo.clone(),
            owner,
            cpu,
        )?;
        if let Ok(mut s) = stats.lock() {
            s.pad_slots = pad.num_slots();
            s.resident_capacity = warm.pinned_slots();
        }
        let resident_capacity = warm.pinned_slots();
        let host_backed = inner.warm_backed.clone();

        let pack = Arc::new(pack);
        let pad = Arc::new(pad);
        let stager_dead = Arc::new(AtomicBool::new(false));
        // The speculative reads the stager lands, for the pipeline thread to
        // list from the pad.
        let (landed_tx, landed_rx) = mpsc::channel();
        let stager = spawn_stager(
            StagerCtx {
                pack: pack.clone(),
                warm: warm.clone(),
                pad: pad.clone(),
                residency: places.clone(),
                clock: dispatch.clock.clone(),
                ring: dispatch.ring.clone(),
                abort: dispatch.abort.clone(),
                consumed: dispatch.staged.clone(),
                landed_ahead: landed_tx,
                stats: stats.clone(),
                rows: num_moe_layers,
                n_experts: experts_per_layer,
                residents,
                resident_capacity,
                host_backed,
                paged_images: paged,
                paged_members,
            },
            stager_dead.clone(),
        )?;

        // The elastic tier (`docs/elastic_warm_tier.md`): pageable chunks grown
        // while this process is quiet and given back when it pages. The pinned
        // part is never touched. A cache holding every expert in VRAM has no
        // tier to move.
        let elastic = if all_resident {
            None
        } else {
            let paged = warm.paged();
            let expert_budget =
                total_physical_ram().map_or(0, |t| host_ram_budget(t).expert_pinned_budget_bytes);
            let pinned_bytes = (warm.pinned_slots() * pad.stride()) as u64;
            let max_chunks = max_chunks(
                order.len(),
                warm.pinned_slots(),
                paged.per_chunk(),
                paged.chunk_bytes(),
                expert_budget,
                pinned_bytes,
            );
            if let Ok(mut s) = stats.lock() {
                s.warm_paged_chunks = paged.chunks();
                s.warm_paged_bytes = paged.held_bytes();
            }
            Some(spawn_controller(ControllerCtx {
                warm: warm.clone(),
                pack,
                order,
                stager: stager.clone(),
                clock: dispatch.clock.clone(),
                device: cuda_dev_ref.clone(),
                stats: stats.clone(),
                max_chunks,
            })?)
        };

        let trace = Arc::new(Mutex::new(RoutingTrace::new()));
        let pinned_slots = pinned_layer_count(num_moe_layers) * experts_per_layer;
        let state = PipelineState::new(
            inner,
            device.clone(),
            copy_stream,
            places,
            dispatch.clock.clone(),
            dispatch.ring.clone(),
            dispatch.abort.clone(),
            dispatch.fault.clone(),
            dispatch.pass.clone(),
            dispatch.served.clone(),
            stager.clone(),
            landed_rx,
            Arc::new(layer_geometries),
            all_resident,
            promo,
            link_rate,
            stats.clone(),
            trace.clone(),
            dispatch.votes.clone(),
            dispatch.read_ahead.clone(),
        )?;
        let pipeline_dead = Arc::new(AtomicBool::new(false));
        let tx = spawn_pipeline_thread(state, pipeline_dead.clone());

        Ok(Self {
            _elastic: elastic,
            tx,
            stager,
            dispatch,
            pipeline_dead,
            stager_dead,
            _warm: warm,
            _pad: pad,
            can_fault,
            cpu_faults: AtomicU32::new(0),
            device: cuda_dev_ref.clone(),
            stats,
            trace,
            pinned_slots,
            serving_floor_slots: floor,
        })
    }

    // Public API

    /// Live VRAM bytes held by resident expert slots (`occupied_slots ×
    /// slot_size`) — the model's **time-varying** MoE weight footprint. Rises as
    /// experts are promoted into VRAM and falls as they are evicted or the zone
    /// concedes ground. Read from the shared stats gauge (seeded at
    /// construction, refreshed by the pipeline thread each routed layer). `0` on non-CUDA / no-expert models.
    pub fn resident_vram_bytes(&self) -> usize {
        PipelineStats::snapshot(&self.stats).resident_vram_bytes
    }
}
