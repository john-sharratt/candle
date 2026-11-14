//! The gap's state from one gap to the next (`docs/kv_quantize_in_the_gap.md`
//! §3): the device state, the code tables, the recorded shares, the one
//! outstanding gap's table and lease, and the deferred blocks waiting to lead
//! the next table.
//!
//! A gap is two calls, with the scheduler's own arena creation and compaction
//! between them:
//!
//! 1. [`GapCycle::catch_up`] — if the outstanding gap has completed, put its
//!    converted blocks in place, record the shares, keep its deferred blocks,
//!    and return its lease to the pools.
//! 2. [`GapCycle::quantize`] — if no gap is outstanding, build the table
//!    (the retried blocks first, in groups of their own, then every sequence's
//!    complete blocks past its cursor, cut to what one gap holds), size its
//!    demand, lease it from the pools, and launch.
//!
//! A gap that finds one outstanding quantizes nothing and takes no block, so
//! no cursor moves past a block no gap will convert.

use std::ops::Range;
use std::time::{Duration, Instant};

use candle::{CudaDevice, Device, Result};

use super::super::backing::ChunkedKvBacking;
use super::super::compress::policy_selection;
use super::super::compression_policy::CompressionPolicy;
use super::super::gid_pool::ChunkGid;
use super::super::head_gids::ChunkBands;
use super::super::size_class::{class_for_format, SizeClass};
use super::super::types::ChunkWindow;
use super::codes::CodeTables;
use super::demand::{gap_demand, GapKind, PeakShares, PerClass};
use super::install::{catch_up, requeue_at, window_of};
use super::launch::{GapDevice, GapGroup, GapJob, MAX_GAP_JOBS};
use super::lease::Lease;
use super::lease_alloc::{lease_demand, LeaseRefusal};
use super::summary::BANDS_PER_HEAD;
use super::table::{gap_take, seal_take};
use crate::kv_cache::KvFormat;

/// The gap launched and not yet caught up.
struct Outstanding {
    jobs: Vec<GapJob>,
    /// Each group's policy and its run of `jobs`.
    groups: Vec<(CompressionPolicy, Range<usize>)>,
    lease: Lease<ChunkGid>,
    kind: GapKind,
}

/// One live sequence a gap may take blocks of, and the policy each of its
/// layers seals under — its conversation's, through the layer's cap. A
/// sequence left native is not handed to the gap at all.
#[derive(Debug, Clone)]
pub struct GapSeq {
    pub seq: usize,
    pub layer_policy: Vec<CompressionPolicy>,
}

/// What one seal's gaps did, summed over its catch-ups — the outstanding
/// gap's it waited for among them.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct GapSealed {
    /// The seal's own jobs launched, over `gaps` gaps.
    pub jobs: usize,
    pub gaps: usize,
    /// Blocks quantized in place, and rows discarded as stale.
    pub installed: usize,
    pub stale: usize,
    /// Blocks deferred at the last catch-up, waiting for a later gap.
    pub deferred: usize,
    /// The gaps' time on the stream.
    pub device_ms: f64,
}

impl GapSealed {
    fn add(&mut self, c: &GapCatchUp) {
        self.installed += c.installed;
        self.stale += c.stale;
        self.deferred = c.deferred;
        self.device_ms += c.device_ms;
    }
}

/// What one catch-up did.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct GapCatchUp {
    /// Whether a gap was outstanding and has completed: false while it still
    /// runs, and when there was none.
    pub caught_up: bool,
    /// Blocks quantized in place.
    pub installed: usize,
    /// Rows discarded: the native chunk was gone.
    pub stale: usize,
    /// Blocks found at another index than the one they were taken at.
    pub moved: usize,
    /// Deferred blocks kept to lead the next table.
    pub deferred: usize,
    /// Whether the claim met a code its tables did not cover.
    pub errored: bool,
    /// The caught-up gap's time on the stream, from its first upload to its
    /// completion word.
    pub device_ms: f64,
    /// Where the call's host time went.
    pub host: CatchUpHost,
}

/// The host time of one catch-up, by phase.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CatchUpHost {
    /// The summary's rows read, each job's chunk found, its slots taken.
    pub read: Duration,
    /// The installed blocks' records built and their arenas resolved.
    pub records: Duration,
    /// The block tables written and their headers uploaded.
    pub install: Duration,
}

/// What one quantize did.
#[derive(Debug, Default)]
pub struct GapQuantize {
    /// The gap is still outstanding: nothing was taken.
    pub outstanding: bool,
    /// Jobs launched — one per block per layer — and how many lead as retries.
    pub jobs: usize,
    pub retried: usize,
    /// Slots each class's lease came up short of its demand, with why.
    pub short: PerClass<usize>,
    pub refusals: Vec<LeaseRefusal>,
    /// Where the call's host time went.
    pub host: QuantizeHost,
}

/// The host time of one quantize, by phase.
#[derive(Debug, Default, Clone, Copy)]
pub struct QuantizeHost {
    /// Building the gap's state, on the first gap of a session.
    pub setup: Duration,
    /// The table: retries re-read, blocks taken, jobs grouped.
    pub table: Duration,
    /// Demand sized and leased from the pools.
    pub lease: Duration,
    /// The host images, the uploads and the launches.
    pub launch: Duration,
}

/// The gap's state, kept by the scheduler from gap to gap.
pub struct GapCycle {
    device: GapDevice,
    codes: CodeTables,
    shares: PeakShares,
    outstanding: Option<Outstanding>,
    /// Deferred blocks still standing, with the policy they are claimed under.
    retry: Vec<(CompressionPolicy, GapJob)>,
    /// Their exact demand per class — of all but the last `unsized_retry`,
    /// whose demand no summary has reported: re-queued blocks, and blocks a
    /// refused launch handed back. Those are sized by share like new blocks.
    retried_demand: PerClass<usize>,
    unsized_retry: usize,
    seq: u64,
    max_jobs: usize,
}

impl GapCycle {
    /// The gap's state for `backings` — one per layer, over one set of pools —
    /// with tables of at most `max_jobs` jobs (one per block per layer).
    pub fn new(backings: &[ChunkedKvBacking], max_jobs: usize) -> Result<Self> {
        let first = backings
            .first()
            .ok_or_else(|| candle::Error::Msg("gap cycle: no layers".into()))?;
        let Device::Cuda(dev) = first.device() else {
            candle::bail!("gap cycle: the gap runs on a CUDA device");
        };
        let dev: &CudaDevice = dev;
        let max_jobs = max_jobs.min(MAX_GAP_JOBS);
        Ok(Self {
            device: GapDevice::new(dev, first.n_kv_head(), first.head_dim(), max_jobs)?,
            codes: CodeTables::new(first.elems_per_chunk())?,
            shares: PeakShares::default(),
            outstanding: None,
            retry: Vec::new(),
            retried_demand: [0; SizeClass::COUNT],
            unsized_retry: 0,
            seq: 0,
            max_jobs,
        })
    }

    /// Whether a gap is outstanding.
    pub fn is_outstanding(&self) -> bool {
        self.outstanding.is_some()
    }

    /// Wait for the outstanding gap's device work, so the next catch-up finds
    /// it complete — for a settle before figures are read. The scheduler
    /// never waits: a gap still running is caught up at a later gap.
    pub fn wait(&self) -> Result<()> {
        self.device.wait()
    }

    /// Put the outstanding gap's converted blocks in place, once it has
    /// completed (§3.5). Nothing when no gap is outstanding or it still runs.
    pub fn catch_up(&mut self, backings: &[ChunkedKvBacking]) -> Result<GapCatchUp> {
        if self.outstanding.is_none() || !self.device.poll() {
            return Ok(GapCatchUp::default());
        }
        // The forwards that read what this installs queue behind the gap.
        self.device.order_compute_after()?;
        let Some(finished) = self.device.finished() else {
            return Ok(GapCatchUp::default());
        };
        let device_ms = finished.device_ms();
        let Outstanding {
            jobs,
            groups,
            mut lease,
            kind,
        } = self.outstanding.take().expect("checked above");
        let refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
        let gap_groups: Vec<GapGroup<'_>> = groups
            .iter()
            .map(|(p, r)| GapGroup {
                policy: p,
                jobs: r.clone(),
            })
            .collect();
        let done = catch_up(&finished, jobs, &gap_groups, &refs, &mut lease, &self.codes)?;
        self.shares.record(kind, &done.claims);
        self.unsized_retry = done.requeue.len();
        self.retry = done
            .retry
            .into_iter()
            .chain(done.requeue)
            .map(|(g, job)| (groups[g].0, job))
            .collect();
        self.retried_demand = done.retried_demand;
        Ok(GapCatchUp {
            caught_up: true,
            installed: done.installed,
            stale: done.stale,
            moved: done.moved,
            deferred: self.retry.len(),
            errored: done.errored,
            device_ms,
            host: done.host,
        })
    }

    /// Quantize every complete block of `seqs` past its cursor, each sequence
    /// under its own layer policies, with the retried blocks first
    /// (§3.1–§3.4). Nothing when a gap is outstanding.
    pub fn quantize(
        &mut self,
        backings: &[ChunkedKvBacking],
        seqs: &[GapSeq],
        kind: GapKind,
    ) -> Result<GapQuantize> {
        if self.outstanding.is_some() {
            return Ok(GapQuantize {
                outstanding: true,
                ..GapQuantize::default()
            });
        }
        if let Some(s) = seqs.iter().find(|s| s.layer_policy.len() != backings.len()) {
            candle::bail!(
                "gap cycle: sequence {} has {} layer policies for {} layers",
                s.seq,
                s.layer_policy.len(),
                backings.len()
            );
        }
        let n_layers = backings.len();
        let t_table = Instant::now();
        let retries = self.take_retries(backings)?;

        // ── Every sequence's complete blocks, cut to the gap's room ──
        let mut room_blocks = self.max_jobs.saturating_sub(retries.jobs.len()) / n_layers.max(1);
        let mut taken: Vec<(&GapSeq, Range<usize>)> = Vec::new();
        for s in seqs {
            if room_blocks == 0 {
                break;
            }
            let range = gap_take(backings, s.seq, room_blocks);
            room_blocks -= range.len();
            if !range.is_empty() {
                taken.push((s, range));
            }
        }
        let mut fresh: Vec<(CompressionPolicy, GapJob)> = Vec::new();
        for (layer, backing) in backings.iter().enumerate() {
            let state = backing
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (s, range) in &taken {
                let seq = state
                    .sequences
                    .get(s.seq)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        candle::Error::Msg(format!("gap cycle: sequence {} gone", s.seq))
                    })?;
                for block in range.clone() {
                    let w = &seq.chunks_slice()[block];
                    fresh.push((s.layer_policy[layer], job_for(layer, s.seq, block, w)));
                }
            }
        }
        self.launch_table(backings, retries, fresh, kind, t_table)
    }

    /// Quantize sequence `seq`'s blocks of `range` now, for its seal: the
    /// writer chunk and every partial chunk with the rest, since the sequence
    /// takes no more writes. Waits for the outstanding gap and catches it up
    /// first, then runs gaps — each waited for and caught up — until every
    /// native block of the range is in place, or two in a row place nothing:
    /// what the ground cannot cover stays native, for the seal to store as it
    /// stands. The deferred blocks of earlier gaps lead each table, as they
    /// lead any gap's.
    pub fn seal(
        &mut self,
        backings: &[ChunkedKvBacking],
        seq: &GapSeq,
        range: Range<usize>,
    ) -> Result<GapSealed> {
        if seq.layer_policy.len() != backings.len() {
            candle::bail!(
                "gap seal: sequence {} has {} layer policies for {} layers",
                seq.seq,
                seq.layer_policy.len(),
                backings.len()
            );
        }
        let mut out = GapSealed::default();
        let mut idle = 0;
        loop {
            if self.outstanding.is_some() {
                self.wait()?;
                out.add(&self.catch_up(backings)?);
            }
            let t_table = Instant::now();
            let retries = self.take_retries(backings)?;
            let room = self.max_jobs.saturating_sub(retries.jobs.len());
            let Some(blocks) = seal_take(backings, seq.seq, range.clone()) else {
                self.put_back(retries);
                return Ok(out);
            };
            let mut fresh: Vec<(CompressionPolicy, GapJob)> = Vec::new();
            for (layer, (backing, blocks)) in backings.iter().zip(&blocks).enumerate() {
                let state = backing
                    .state
                    .read()
                    .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
                let Some(s) = state.sequences.get(seq.seq).and_then(Option::as_ref) else {
                    continue;
                };
                for &block in blocks.iter().take(room.saturating_sub(fresh.len())) {
                    let w = &s.chunks_slice()[block];
                    fresh.push((seq.layer_policy[layer], job_for(layer, seq.seq, block, w)));
                }
            }
            if fresh.is_empty() {
                self.put_back(retries);
                return Ok(out);
            }
            out.jobs += fresh.len();
            out.gaps += 1;
            // A seal's blocks are its turn's last: decode's, for the shares.
            self.launch_table(backings, retries, fresh, GapKind::Decode, t_table)?;
            self.wait()?;
            let c = self.catch_up(backings)?;
            out.add(&c);
            idle = if c.installed == 0 { idle + 1 } else { 0 };
            if idle >= SEAL_STALLED_GAPS {
                return Ok(out);
            }
        }
    }

    /// The deferred blocks standing, each re-read at its index — the
    /// compaction that ran since the catch-up may have moved its bands, and
    /// moves no index — with their demand. One no longer a native block of its
    /// window behind its cursor is dropped.
    fn take_retries(&mut self, backings: &[ChunkedKvBacking]) -> Result<Retries> {
        let demand = std::mem::replace(&mut self.retried_demand, [0; SizeClass::COUNT]);
        let unsized_jobs = std::mem::take(&mut self.unsized_retry);
        let refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
        let mut jobs = Vec::with_capacity(self.retry.len());
        for (p, mut job) in std::mem::take(&mut self.retry) {
            if let Some(bands) = requeue_at(&refs, &job)? {
                job.bands = bands;
                jobs.push((p, job));
            }
        }
        Ok(Retries {
            jobs,
            demand,
            unsized_jobs,
        })
    }

    /// Retries a table was not launched for, kept for the next.
    fn put_back(&mut self, retries: Retries) {
        self.retry = retries.jobs;
        self.retried_demand = retries.demand;
        self.unsized_retry = retries.unsized_jobs;
    }

    /// Lay the table out — the retries first, in groups of their own, then
    /// `fresh` — size its demand, lease it and launch it. Nothing when the
    /// table is empty.
    fn launch_table(
        &mut self,
        backings: &[ChunkedKvBacking],
        retries: Retries,
        fresh: Vec<(CompressionPolicy, GapJob)>,
        kind: GapKind,
        t_table: Instant,
    ) -> Result<GapQuantize> {
        let Retries {
            jobs: retry,
            demand: retried_demand,
            unsized_jobs: unsized_retry,
        } = retries;
        let mut jobs: Vec<GapJob> = Vec::new();
        let mut groups: Vec<(CompressionPolicy, Range<usize>)> = Vec::new();
        let n_retry = retry.len();
        push_grouped(&mut jobs, &mut groups, retry);
        let n_fresh = fresh.len();
        push_grouped(&mut jobs, &mut groups, fresh);
        if jobs.is_empty() {
            return Ok(GapQuantize::default());
        }

        // ── Demand, lease, launch ──
        let mut host = QuantizeHost {
            table: t_table.elapsed(),
            ..QuantizeHost::default()
        };
        let t_lease = Instant::now();
        let n_kv_head = backings[0].n_kv_head();
        let new_bands = (n_fresh + unsized_retry.min(n_retry)) * n_kv_head * BANDS_PER_HEAD;
        let reach = reachable(&groups, backings[0].elems_per_chunk())?;
        let peak = self.shares.peak(kind);
        let demand = gap_demand(&retried_demand, new_bands, peak.as_ref(), &reach);
        let leased = lease_demand(&backings[0], &demand)?;
        let mut lease = leased.lease;
        host.lease = t_lease.elapsed();
        let t_launch = Instant::now();
        let launched = {
            let refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
            let gap_groups: Vec<GapGroup<'_>> = groups
                .iter()
                .map(|(p, r)| GapGroup {
                    policy: p,
                    jobs: r.clone(),
                })
                .collect();
            lease.stack_image().and_then(|image| {
                self.device
                    .launch(&refs, &jobs, &gap_groups, &image, self.seq)
            })
        };
        if let Err(e) = launched {
            // Nothing ran: every job's block is still native and behind its
            // cursor, so all of them wait as retries.
            drop(lease.release());
            let policies = groups.iter().flat_map(|(p, r)| r.clone().map(move |_| *p));
            self.retry = policies.zip(jobs).collect();
            // Their demand is not known to the slot: the new blocks among them
            // were never claimed. The next gap sizes them all by share.
            self.retried_demand = [0; SizeClass::COUNT];
            self.unsized_retry = self.retry.len();
            return Err(e);
        }
        host.launch = t_launch.elapsed();
        self.seq = self.seq.wrapping_add(1);
        let n_jobs = jobs.len();
        self.outstanding = Some(Outstanding {
            jobs,
            groups,
            lease,
            kind,
        });
        Ok(GapQuantize {
            outstanding: false,
            jobs: n_jobs,
            retried: n_retry,
            short: leased.short,
            refusals: leased.refusals,
            host,
        })
    }
}

/// Gaps in a row a seal lets place nothing before it leaves the rest native:
/// one is a retry finding its ground, a second the same blocks refused again.
const SEAL_STALLED_GAPS: usize = 2;

/// The deferred blocks a table leads with, and their demand.
struct Retries {
    jobs: Vec<(CompressionPolicy, GapJob)>,
    demand: PerClass<usize>,
    /// How many of the last of `jobs` no summary has sized.
    unsized_jobs: usize,
}

/// The job that converts `block` of sequence `seq` on `layer`, over its valid
/// tokens.
fn job_for(layer: usize, seq: usize, block: usize, w: &ChunkWindow) -> GapJob {
    GapJob {
        layer,
        seq,
        block,
        bands: ChunkBands::from_window(w),
        valid_range: window_of(w),
    }
}

/// Append `jobs` to the table in one group per policy, in the order the
/// policies first appear. Policies are told apart by level and overrides.
fn push_grouped(
    table: &mut Vec<GapJob>,
    groups: &mut Vec<(CompressionPolicy, Range<usize>)>,
    jobs: Vec<(CompressionPolicy, GapJob)>,
) {
    let mut order: Vec<CompressionPolicy> = Vec::new();
    for (p, _) in &jobs {
        if !order.iter().any(|q| same_policy(q, p)) {
            order.push(*p);
        }
    }
    let mut jobs: Vec<Option<(CompressionPolicy, GapJob)>> = jobs.into_iter().map(Some).collect();
    for p in order {
        let start = table.len();
        for slot in jobs.iter_mut() {
            if slot.as_ref().is_some_and(|(q, _)| same_policy(q, &p)) {
                table.push(slot.take().expect("checked").1);
            }
        }
        groups.push((p, start..table.len()));
    }
}

/// Whether two policies select alike: the same level, the same threshold
/// factors and the same overrides — what one selection launch takes.
fn same_policy(a: &CompressionPolicy, b: &CompressionPolicy) -> bool {
    let factors = |p: &CompressionPolicy| {
        [
            p.k_hi_error_threshold_factor,
            p.k_low_error_threshold_factor,
            p.v_hi_error_threshold_factor,
            p.v_low_error_threshold_factor,
        ]
        .map(f32::to_bits)
    };
    a.compression_level == b.compression_level
        && factors(a) == factors(b)
        && a.override_k_quant == b.override_k_quant
        && a.override_v_quant == b.override_v_quant
}

/// The classes any group's candidates or overrides reach.
fn reachable(
    groups: &[(CompressionPolicy, Range<usize>)],
    elems_per_chunk: usize,
) -> Result<PerClass<bool>> {
    let mut reach = [false; SizeClass::COUNT];
    for (p, _) in groups {
        let sel = policy_selection(p)?;
        let formats = sel
            .k
            .iter()
            .chain(&sel.v)
            .filter_map(|f| f.to_quant_format())
            .chain(p.override_k_quant)
            .chain(p.override_v_quant);
        for q in formats {
            if let Some(c) = class_for_format(KvFormat::Quantized(q), elems_per_chunk) {
                reach[c.index()] = true;
            }
        }
    }
    Ok(reach)
}
