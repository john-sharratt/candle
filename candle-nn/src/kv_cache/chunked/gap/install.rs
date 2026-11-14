//! The catch-up (`docs/kv_quantize_in_the_gap.md` §3.5): at the head of a gap,
//! the previous gap's converted blocks put in place of their native chunks.
//!
//! The summary is read once its completion word is in ([`Finished`]). For every
//! job the host finds the native chunk the gap converted: at its index, or —
//! when a front eviction or a splice moved it while the gap ran — where it
//! moved to, found by identity. A chunk truncated or evicted away is gone, and
//! its row is discarded. A converted block that still stands takes its slots
//! out of the lease, block by block, and becomes a
//! quantized chunk laid out as today's seal lays it out; every such chunk's
//! record is built in one batch, and each layer's chunks are swapped in under
//! one write of its block table, their headers re-serialised in one upload per
//! sequence and the writer boundary moved past them. A deferred block that
//! still matches goes back to the caller to lead the next table, with its exact
//! demand. Every slot left in the lease — past every class's demand, given to
//! a deferred block, or named by a stale row — goes back to its pool, and the
//! native bands go with the jobs that held them.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use candle::Result;

use super::super::backing::ChunkedKvBacking;
use super::super::gid_pool::ChunkGid;
use super::super::head_gids::{ChunkBands, HeadGids};
use super::super::meta_pool::{ChunkRecordSrc, MetaGid};
use super::super::size_class::SizeClass;
use super::super::types::{ChunkWindow, QSign, CHUNK_SIZE};
use super::assemble::{block_state, take_block, BlockState};
use super::codes::CodeTables;
use super::cycle::CatchUpHost;
use super::demand::{count_bands, PerClass};
use super::launch::{GapGroup, GapJob};
use super::lease::Lease;
use super::summary::{Finished, HeadRow};

/// What one catch-up did.
#[derive(Debug)]
pub(crate) struct CatchUp {
    /// Blocks now quantized in place.
    pub installed: usize,
    /// Rows discarded because the native chunk the gap converted is gone from
    /// its sequence — deferred or not.
    pub stale: usize,
    /// Jobs whose native chunk had moved to another index while the gap ran
    /// (a front eviction, a splice), found there and handled there.
    pub moved: usize,
    /// Deferred blocks still in place, each with the index of the group it was
    /// claimed under: they lead the next table, in groups of their own.
    pub retry: Vec<(usize, GapJob)>,
    /// The exact per-class demand of `retry`.
    pub retried_demand: PerClass<usize>,
    /// Blocks whose native chunk was replaced at its index by another native
    /// chunk while the gap ran — a compaction relocating its bands — and sits
    /// behind the cursor: re-queued with the bands it holds now, to be
    /// quantized afresh, never installed from what the gap read. Their demand
    /// is not known to the slot.
    pub requeue: Vec<(usize, GapJob)>,
    /// Every band of the gap's table counted by class, deferred or not: what
    /// the shares are recorded from.
    pub claims: PerClass<usize>,
    /// Whether the claim met a code its tables did not cover. The jobs it met
    /// it in are among the deferred.
    pub errored: bool,
    /// Where the host time went.
    pub host: CatchUpHost,
}

/// One converted block on its way in.
struct Install {
    job: usize,
    gids: HeadGids,
    state: BlockState,
    /// Its provenance `sign(Q)`, read from the R16 bands it replaces.
    q_sign: Option<QSign>,
}

/// The group `job` was claimed under.
fn group_of(groups: &[GapGroup<'_>], job: usize) -> Result<usize> {
    groups
        .iter()
        .position(|g| g.jobs.contains(&job))
        .ok_or_else(|| candle::Error::Msg(format!("gap install: job {job} is in no group")))
}

/// Where the native chunk `job` converted stands now: at its index, or — when
/// a front eviction or a splice moved it while the gap ran — wherever it moved
/// to in its sequence, found by identity. `None` when it is gone: truncated,
/// evicted, or its sequence freed. The search runs only for a job whose index
/// no longer holds it, which a quiet gap never has.
fn locate(chunks: &[ChunkWindow], job: &GapJob) -> Option<usize> {
    let at = |i: usize| {
        chunks
            .get(i)
            .is_some_and(|c| c.gids.is_same_alloc(&job.bands.gids))
    };
    if at(job.block) {
        return Some(job.block);
    }
    (0..chunks.len()).find(|&i| at(i))
}

/// The bands of the native chunk now at `job`'s index, when it is a complete
/// native block behind its sequence's cursor that no gap will take again: what
/// a compaction leaves when it relocates a block's bands while a gap reads the
/// old ones. A chunk past the cursor is the next gap's anyway; a quantized or
/// partial one is no block to quantize.
pub(super) fn requeue_at(
    backings: &[&ChunkedKvBacking],
    job: &GapJob,
) -> Result<Option<ChunkBands>> {
    let backing = backings.get(job.layer).ok_or_else(|| {
        candle::Error::Msg(format!("gap install: job layer {} out of range", job.layer))
    })?;
    let state = backing
        .state
        .read()
        .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
    Ok(state
        .sequences
        .get(job.seq)
        .and_then(Option::as_ref)
        .filter(|s| job.block < s.gap_cursor())
        .and_then(|s| s.chunks_slice().get(job.block))
        .filter(|c| c.takes_writes() && c.usage as usize == CHUNK_SIZE)
        .map(ChunkBands::from_window))
}

/// [`locate`] in `job`'s layer.
fn locate_in(backings: &[&ChunkedKvBacking], job: &GapJob) -> Result<Option<usize>> {
    let backing = backings.get(job.layer).ok_or_else(|| {
        candle::Error::Msg(format!("gap install: job layer {} out of range", job.layer))
    })?;
    let state = backing
        .state
        .read()
        .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
    Ok(state
        .sequences
        .get(job.seq)
        .and_then(Option::as_ref)
        .and_then(|s| locate(s.chunks_slice(), job)))
}

/// Put the converted blocks of the gap `finished` describes in place, as the
/// module explains. `jobs` and `groups` are the table the gap was launched
/// with, `lease` the lease its stack was built from — emptied here, whatever
/// the outcome.
pub(crate) fn catch_up(
    finished: &Finished<'_>,
    jobs: Vec<GapJob>,
    groups: &[GapGroup<'_>],
    backings: &[&ChunkedKvBacking],
    lease: &mut Lease<ChunkGid>,
    codes: &CodeTables,
) -> Result<CatchUp> {
    let result = catch_up_inner(finished, jobs, groups, backings, lease, codes);
    // Whatever happened above, nothing is leased past the catch-up.
    drop(lease.release());
    result
}

fn catch_up_inner(
    finished: &Finished<'_>,
    jobs: Vec<GapJob>,
    groups: &[GapGroup<'_>],
    backings: &[&ChunkedKvBacking],
    lease: &mut Lease<ChunkGid>,
    codes: &CodeTables,
) -> Result<CatchUp> {
    if finished.jobs() != jobs.len() {
        candle::bail!(
            "gap install: a summary of {} jobs for a table of {}",
            finished.jobs(),
            jobs.len()
        );
    }
    let class_of_fmt = |f: u8| codes.class_of_fmt(f);
    let t_read = Instant::now();
    let mut out = CatchUp {
        installed: 0,
        stale: 0,
        moved: 0,
        retry: Vec::new(),
        retried_demand: [0; SizeClass::COUNT],
        requeue: Vec::new(),
        claims: [0; SizeClass::COUNT],
        errored: finished.errored(),
        host: CatchUpHost::default(),
    };
    let mut installs: Vec<Install> = Vec::new();
    // Deferred jobs whose native chunk still stands, and where it stands now.
    let mut retry: Vec<(usize, usize)> = Vec::new();
    // Jobs whose index now holds another native chunk behind the cursor.
    let mut requeue: Vec<(usize, ChunkBands)> = Vec::new();
    for (j, job) in jobs.iter().enumerate() {
        let rows: Vec<HeadRow> = finished.rows(j)?;
        let deferred = finished.deferred(j)?;
        // Every band the claim placed in a class, deferred or not. A job the
        // claim refused for a code its tables do not cover has a band with no
        // class: it is deferred, counted nowhere, and not retried — it would be
        // refused again.
        let counted = count_bands(&rows, class_of_fmt).ok();
        if let Some(counted) = counted {
            for (c, n) in out.claims.iter_mut().zip(counted) {
                *c += n;
            }
        }
        let Some(at) = locate_in(backings, job)? else {
            match requeue_at(backings, job)? {
                Some(bands) => requeue.push((j, bands)),
                None => out.stale += 1,
            }
            continue;
        };
        if at != job.block {
            out.moved += 1;
        }
        if deferred {
            if let Some(demand) = counted {
                for (c, n) in out.retried_demand.iter_mut().zip(demand) {
                    *c += n;
                }
                retry.push((j, at));
            }
            continue;
        }
        let gids = HeadGids::from_vec(take_block(lease, &rows, class_of_fmt)?);
        let policy = groups[group_of(groups, j)?].policy;
        let state = block_state(
            &rows,
            backings[job.layer].head_dim(),
            policy.override_k_quant.is_some(),
            policy.override_v_quant.is_some(),
            |f| codes.tag_of_fmt(f),
        )?;
        installs.push(Install {
            job: j,
            gids,
            state,
            q_sign: finished.q_sign(j)?,
        });
    }

    out.host.read = t_read.elapsed();
    // Every installed block's record, in one batch.
    if let Some(first) = installs.first() {
        let t_records = Instant::now();
        let backing = backings[jobs[first.job].layer];
        let needed: HashSet<usize> = installs
            .iter()
            .flat_map(|i| i.gids.as_slice().iter().map(ChunkGid::arena_idx))
            .collect();
        let band_info = backing.resolve_arena_info_for(&needed)?;
        let srcs: Vec<ChunkRecordSrc<'_>> = installs
            .iter()
            .map(|i| ChunkRecordSrc {
                gids: &i.gids,
                k_pal: &i.state.k_pal,
                v_pal: &i.state.v_pal,
                k_scale: &i.state.k_scale,
                v_scale: &i.state.v_scale,
                k_fmt: &i.state.k_fmt,
                v_fmt: &i.state.v_fmt,
            })
            .collect();
        let records = backing.build_meta_records(&srcs, &band_info)?;
        drop(srcs);

        // The headers name the bands' arenas and the records'.
        let mut header_arenas = needed;
        header_arenas.extend(records.iter().flatten().map(|r| r.gid().arena_idx()));
        let header_info = backing.resolve_arena_info_for(&header_arenas)?;
        out.host.records = t_records.elapsed();
        let t_install = Instant::now();

        // Per layer, one write of its block table.
        let mut by_layer: Vec<Vec<(Install, Option<MetaGid>)>> =
            (0..backings.len()).map(|_| Vec::new()).collect();
        for (install, record) in installs.into_iter().zip(records) {
            by_layer[jobs[install.job].layer].push((install, record));
        }
        for (layer, blocks) in by_layer.into_iter().enumerate() {
            if blocks.is_empty() {
                continue;
            }
            let backing = backings[layer];
            let mut state = backing
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let mut touched: Vec<(usize, Vec<usize>)> = Vec::new();
            for (install, record) in blocks {
                let job = &jobs[install.job];
                let Some(seq) = state.sequences.get_mut(job.seq).and_then(Option::as_mut) else {
                    out.stale += 1;
                    continue;
                };
                let Some(block) = locate(seq.chunks_slice(), job) else {
                    // Its new slots go back to the pool as `install` drops.
                    out.stale += 1;
                    continue;
                };
                let BlockState {
                    k_pal,
                    v_pal,
                    k_scale,
                    v_scale,
                    k_fmt,
                    v_fmt,
                } = install.state;
                seq.set_block_gids(
                    block,
                    install.gids,
                    Arc::new(k_pal),
                    Arc::new(v_pal),
                    Arc::new(k_scale),
                    Arc::new(v_scale),
                    Arc::new(k_fmt),
                    Arc::new(v_fmt),
                );
                if let Some(cw) = seq.chunk_at_mut(block) {
                    cw.meta = record;
                    // The bands that held its Q are going; their signs stay.
                    cw.q_sign = install.q_sign;
                }
                match touched.iter_mut().find(|(s, _)| *s == job.seq) {
                    Some((_, blocks)) => blocks.push(block),
                    None => touched.push((job.seq, vec![block])),
                }
                out.installed += 1;
            }
            for (s, blocks) in touched {
                let Some(seq) = state.sequences.get_mut(s).and_then(Option::as_mut) else {
                    continue;
                };
                seq.update_gpu_chunks_bulk(
                    &blocks,
                    backing.n_kv_head(),
                    backing.head_dim(),
                    &header_info,
                )?;
                // A quantized chunk takes no writes: below the writer chunk it
                // is sealed history, and the boundary moves past it.
                seq.seal_unwritable_writer_region();
            }
        }
        out.host.install = t_install.elapsed();
    }

    // The deferred blocks still standing lead the next table, at the index
    // they stand at now; every other job drops here, and with it the last
    // hold on its native bands.
    let groups_of: Vec<usize> = retry
        .iter()
        .map(|&(j, _)| group_of(groups, j))
        .collect::<Result<_>>()?;
    let requeue_groups: Vec<usize> = requeue
        .iter()
        .map(|(j, _)| group_of(groups, *j))
        .collect::<Result<_>>()?;
    let mut retry = retry.into_iter().zip(groups_of).peekable();
    let mut requeue = requeue.into_iter().zip(requeue_groups).peekable();
    for (j, mut job) in jobs.into_iter().enumerate() {
        if retry.peek().is_some_and(|&((r, _), _)| r == j) {
            let ((_, at), g) = retry.next().expect("peeked");
            job.block = at;
            out.retry.push((g, job));
        } else if requeue.peek().is_some_and(|((r, _), _)| *r == j) {
            let ((_, bands), g) = requeue.next().expect("peeked");
            job.bands = bands;
            out.requeue.push((g, job));
        }
    }
    Ok(out)
}
