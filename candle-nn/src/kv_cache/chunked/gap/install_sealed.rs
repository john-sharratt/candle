//! A gap's converted blocks put into a sealed residence's chunks
//! (`docs/kv_quantize_in_the_gap.md` §5): the catch-up for a table whose jobs
//! name the chunks of a hot copy rather than a live sequence. A turn whose
//! seal gap the ground could not cover is stored with native chunks; a later
//! gap converts them and hands back a copy of the residence's sequences with
//! those chunks quantized — the bands, formats, scales, maps, record and
//! signs a live install gives a chunk — for the caller to swap in over the
//! hot copy it read, by lineage. A deferred block stays native in the copy,
//! for the next gap to take.

use std::collections::HashSet;
use std::sync::Arc;

use candle::Result;

use super::super::backing::ChunkedKvBacking;
use super::super::gid_pool::ChunkGid;
use super::super::head_gids::HeadGids;
use super::super::meta_pool::ChunkRecordSrc;
use super::super::size_class::SizeClass;
use super::super::types::SealedSequence;
use super::assemble::{block_state, take_block, BlockState};
use super::codes::CodeTables;
use super::demand::{count_bands, PerClass};
use super::launch::{GapGroup, GapJob};
use super::lease::Lease;
use super::summary::Finished;

/// What one sealed catch-up did.
#[derive(Debug)]
pub(crate) struct SealedCatchUp {
    /// The residence's sequences, one a layer, with every installed block
    /// quantized.
    pub seqs: Vec<SealedSequence>,
    pub installed: usize,
    /// Blocks the claim deferred: native still in `seqs`.
    pub deferred: usize,
    /// Every band of the table counted by class, for the shares.
    pub claims: PerClass<usize>,
}

/// Put the converted blocks of the gap `finished` describes into a copy of
/// `hot` — one sequence a layer, each job naming `(layer, block)` of it.
/// `lease` is emptied here, whatever the outcome.
pub(crate) fn catch_up_sealed(
    finished: &Finished<'_>,
    jobs: &[GapJob],
    groups: &[GapGroup<'_>],
    backings: &[&ChunkedKvBacking],
    lease: &mut Lease<ChunkGid>,
    codes: &CodeTables,
    hot: &[SealedSequence],
) -> Result<SealedCatchUp> {
    let result = catch_up_inner(finished, jobs, groups, backings, lease, codes, hot);
    drop(lease.release());
    result
}

fn catch_up_inner(
    finished: &Finished<'_>,
    jobs: &[GapJob],
    groups: &[GapGroup<'_>],
    backings: &[&ChunkedKvBacking],
    lease: &mut Lease<ChunkGid>,
    codes: &CodeTables,
    hot: &[SealedSequence],
) -> Result<SealedCatchUp> {
    if finished.jobs() != jobs.len() {
        candle::bail!(
            "gap sealed install: a summary of {} jobs for a table of {}",
            finished.jobs(),
            jobs.len()
        );
    }
    let class_of_fmt = |f: u8| codes.class_of_fmt(f);
    let mut out = SealedCatchUp {
        seqs: hot.to_vec(),
        installed: 0,
        deferred: 0,
        claims: [0; SizeClass::COUNT],
    };
    let mut installs: Vec<(usize, HeadGids, BlockState)> = Vec::new();
    for (j, job) in jobs.iter().enumerate() {
        let rows = finished.rows(j)?;
        if let Ok(counted) = count_bands(&rows, class_of_fmt) {
            for (c, n) in out.claims.iter_mut().zip(counted) {
                *c += n;
            }
        }
        if finished.deferred(j)? {
            out.deferred += 1;
            continue;
        }
        let gids = HeadGids::from_vec(take_block(lease, &rows, class_of_fmt)?);
        let policy = groups
            .iter()
            .find(|g| g.jobs.contains(&j))
            .ok_or_else(|| candle::Error::Msg(format!("gap sealed install: job {j} in no group")))?
            .policy;
        let state = block_state(
            &rows,
            backings[job.layer].head_dim(),
            policy.override_k_quant.is_some(),
            policy.override_v_quant.is_some(),
            |f| codes.tag_of_fmt(f),
        )?;
        installs.push((j, gids, state));
    }
    let Some(&(first, ..)) = installs.first() else {
        return Ok(out);
    };
    let backing = backings[jobs[first].layer];
    let needed: HashSet<usize> = installs
        .iter()
        .flat_map(|(_, g, _)| g.as_slice().iter().map(ChunkGid::arena_idx))
        .collect();
    let band_info = backing.resolve_arena_info_for(&needed)?;
    let srcs: Vec<ChunkRecordSrc<'_>> = installs
        .iter()
        .map(|(_, gids, s)| ChunkRecordSrc {
            gids,
            k_pal: &s.k_pal,
            v_pal: &s.v_pal,
            k_scale: &s.k_scale,
            v_scale: &s.v_scale,
            k_fmt: &s.k_fmt,
            v_fmt: &s.v_fmt,
        })
        .collect();
    let records = backing.build_meta_records(&srcs, &band_info)?;
    drop(srcs);
    let elems = backing.elems_per_chunk();
    for ((j, gids, state), record) in installs.into_iter().zip(records) {
        let job = &jobs[j];
        let chunk = out
            .seqs
            .get_mut(job.layer)
            .and_then(|s| s.chunks.get_mut(job.block))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "gap sealed install: layer {} block {} is not in the residence",
                    job.layer, job.block
                ))
            })?;
        let q_sign = finished.q_sign(j)?;
        chunk.gids = gids;
        chunk.k_pal = Arc::new(state.k_pal);
        chunk.v_pal = Arc::new(state.v_pal);
        chunk.k_scale = Arc::new(state.k_scale);
        chunk.v_scale = Arc::new(state.v_scale);
        chunk.k_fmt = Arc::new(state.k_fmt);
        chunk.v_fmt = Arc::new(state.v_fmt);
        chunk.meta = record;
        chunk.q_sign = q_sign.or_else(|| chunk.q_sign.clone());
        chunk.byte_size = chunk.byte_size(elems);
        out.installed += 1;
    }
    Ok(out)
}
