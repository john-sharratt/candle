//! Cold-load chunk planning — partition a turn's records into
//! buffer-sized chunks, with one stripe set per chunk.
//!
//! The cold-load pipeline streams a turn's bytes through a fixed-size
//! pinned scratch ([`ColdLoadStager`]'s buffer, sized to
//! [`super::cold_load::PINNED_PREALLOC_BYTES`]). Each "chunk" here is
//! a subset of the turn's records whose summed `record_size` fits in
//! that buffer; the orchestrator processes one chunk at a time
//! (read → decode → alloc dest blocks → HtoD → kv_migrate) before
//! refilling the buffer for the next chunk.
//!
//! The planner runs **before any I/O** so the orchestrator knows
//! upfront how many chunks to process, and so each chunk is a
//! contiguous prefix of records in disk order — preserving the
//! stripe coalescing that makes the NVMe reads sequential.
//!
//! ## Constraints
//!
//! - A chunk's records are **all from the same source log** (active
//!   or one specific inherited). Crossing sources mid-chunk would
//!   break stripe coalescing (offsets are per-file). Sources are
//!   processed sequentially; one source's chunks are emitted
//!   contiguously, then the next source's.
//! - A chunk's total bytes are `<= buffer_size`. Records are not
//!   split across chunks (records are 4 KiB aligned; the buffer is
//!   sized in MiB; a single record always fits if the buffer is at
//!   least 4 KiB).
//! - Empty turns yield a plan with `chunks.len() == 0` and
//!   `total_bytes == 0`.

use std::collections::{BTreeMap, HashSet};

use super::crc_validator::BadChunkRegistry;
use super::manifest::ChunkLoc;
use super::streams::StreamId;

/// Identifies which log a record's bytes live in. The orchestrator
/// uses this to pick the correct [`super::direct_io::DirectFile`]
/// handle when reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceLog {
    /// The substrate's own active redo log.
    Active,
    /// The `i`-th inherited log (index into
    /// [`SubstratePersistence::inherited`]).
    Inherited(usize),
}

/// One record's on-disk and in-chunk-buffer location, captured at
/// planning time before any I/O happens.
#[derive(Clone, Debug)]
pub struct PlannedRecord {
    pub source: SourceLog,
    pub chunk_idx: u64,
    pub file_offset: u64,
    /// On-disk padded record size — also the length of bytes the
    /// chunked read will land in the buffer for this record.
    pub record_size: u64,
    pub payload_len: u64,
    pub token_count: u64,
    /// Offset within the chunk's buffer where this record's bytes
    /// land. `buf_offset + record_size` is at most the chunk's
    /// `total_bytes`.
    pub buf_offset: usize,
}

/// One coalesced disk stripe within a chunk: adjacent records on
/// disk that read as a single contiguous span. Built from sorted
/// `PlannedRecord`s by walking and merging adjacent
/// `file_offset + record_size == next.file_offset` neighbours.
#[derive(Clone, Copy, Debug)]
pub struct ChunkStripe {
    pub file_offset: u64,
    pub len: usize,
    /// Offset within the chunk's buffer where this stripe lands.
    pub buf_offset: usize,
}

/// One chunked-read batch. The orchestrator reads `stripes` into
/// the stager's buffer (`total_bytes` bytes), decodes the `records`,
/// then HtoDs + migrates them, before moving on to the next chunk.
#[derive(Clone, Debug)]
pub struct ChunkBatch {
    pub source: SourceLog,
    pub stripes: Vec<ChunkStripe>,
    pub records: Vec<PlannedRecord>,
    /// Sum of stripe lengths — also the byte count the chunked
    /// read uploads via HtoD.
    pub total_bytes: usize,
}

/// Complete chunk plan for one cold-load — every record in the
/// turn, partitioned into [`ChunkBatch`]es that each fit within
/// `buffer_size` bytes.
#[derive(Clone, Debug, Default)]
pub struct ChunkedReadPlan {
    pub chunks: Vec<ChunkBatch>,
    /// Sum of every chunk's `total_bytes` — the turn's full
    /// on-disk record footprint.
    pub total_bytes: usize,
    /// Total record count across all chunks. Equals the
    /// `n_layers * chunks_per_layer` of the turn (one record per
    /// layer × per chunk-block).
    pub n_records: usize,
}

/// Build a [`ChunkedReadPlan`] from an active stream's chunk index
/// and an ordered list of inherited chunk indices.
///
/// Walks the active source first — its records win on `chunk_idx`
/// collision via the `seen` set, matching the existing
/// `read_stream_chunks_batched` semantics. Each inherited source then
/// fills indices not present in active or in earlier-inherited
/// sources. Within each source the records are sorted by
/// `file_offset` and partitioned greedily into chunks of
/// `<= buffer_size` bytes. Empty sources contribute no chunks; an
/// empty turn yields an empty plan.
///
/// Chunks flagged in `bad_chunks` are skipped — the background CRC
/// validator has marked them as bit-rot, and the cold-load would
/// fail downstream anyway.
pub fn plan_chunked_read(
    active_chunks: Option<&BTreeMap<u64, ChunkLoc>>,
    inherited_chunks: &[Option<&BTreeMap<u64, ChunkLoc>>],
    stream_id: StreamId,
    buffer_size: usize,
    bad_chunks: &BadChunkRegistry,
) -> ChunkedReadPlan {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut plan = ChunkedReadPlan::default();

    {
        let active_recs = collect_records(
            active_chunks,
            SourceLog::Active,
            stream_id,
            &mut seen,
            bad_chunks,
        );
        partition_and_append(active_recs, buffer_size, &mut plan);
    }

    for (i, chunks) in inherited_chunks.iter().enumerate() {
        let inh_recs = collect_records(
            *chunks,
            SourceLog::Inherited(i),
            stream_id,
            &mut seen,
            bad_chunks,
        );
        partition_and_append(inh_recs, buffer_size, &mut plan);
    }

    plan
}

fn collect_records(
    chunks: Option<&std::collections::BTreeMap<u64, ChunkLoc>>,
    source: SourceLog,
    stream_id: StreamId,
    seen: &mut HashSet<u64>,
    bad_chunks: &BadChunkRegistry,
) -> Vec<RawRecord> {
    let Some(chunks) = chunks else {
        return Vec::new();
    };
    let mut out: Vec<RawRecord> = Vec::with_capacity(chunks.len());
    for (&chunk_idx, loc) in chunks {
        if !seen.insert(chunk_idx) {
            continue;
        }
        if bad_chunks.is_bad((stream_id, chunk_idx)) {
            continue;
        }
        out.push(RawRecord {
            source,
            chunk_idx,
            file_offset: loc.offset,
            record_size: loc.record_size,
            payload_len: loc.payload_len,
            token_count: loc.token_count,
        });
    }
    out.sort_unstable_by_key(|r| r.file_offset);
    out
}

#[derive(Clone, Copy)]
struct RawRecord {
    source: SourceLog,
    chunk_idx: u64,
    file_offset: u64,
    record_size: u64,
    payload_len: u64,
    token_count: u64,
}

/// Greedily pack `records` (sorted by `file_offset`) into chunks of
/// `<= buffer_size` bytes, appending each chunk to `plan`.
///
/// A new chunk starts when adding the next record would overflow
/// `buffer_size`. Within a chunk, stripes are built by coalescing
/// adjacent records — the same logic the legacy
/// `build_stripes` used, but per-chunk so stripe spans never cross
/// chunk boundaries (which the buffer can't represent anyway).
fn partition_and_append(records: Vec<RawRecord>, buffer_size: usize, plan: &mut ChunkedReadPlan) {
    if records.is_empty() {
        return;
    }
    let source = records[0].source;
    let mut cur_records: Vec<PlannedRecord> = Vec::new();
    let mut cur_bytes: usize = 0;

    for raw in records {
        let rec_bytes = raw.record_size as usize;
        debug_assert!(
            rec_bytes <= buffer_size,
            "PlannedRecord chunk_idx {} has record_size {} > buffer_size {}",
            raw.chunk_idx,
            rec_bytes,
            buffer_size,
        );
        if cur_bytes + rec_bytes > buffer_size {
            // Seal the current chunk and start a new one.
            plan.n_records += cur_records.len();
            let batch = seal_chunk(source, std::mem::take(&mut cur_records), cur_bytes);
            plan.total_bytes += batch.total_bytes;
            plan.chunks.push(batch);
            cur_bytes = 0;
        }
        cur_records.push(PlannedRecord {
            source: raw.source,
            chunk_idx: raw.chunk_idx,
            file_offset: raw.file_offset,
            record_size: raw.record_size,
            payload_len: raw.payload_len,
            token_count: raw.token_count,
            buf_offset: cur_bytes,
        });
        cur_bytes += rec_bytes;
    }
    if !cur_records.is_empty() {
        plan.n_records += cur_records.len();
        let batch = seal_chunk(source, cur_records, cur_bytes);
        plan.total_bytes += batch.total_bytes;
        plan.chunks.push(batch);
    }
}

fn seal_chunk(source: SourceLog, records: Vec<PlannedRecord>, total_bytes: usize) -> ChunkBatch {
    let stripes = build_stripes_from_records(&records);
    ChunkBatch {
        source,
        stripes,
        records,
        total_bytes,
    }
}

/// One 1 MiB pipeline unit within a chunk batch — the work unit for
/// the pipelined cold-load loop. A unit is fully contained within a
/// stripe (so its `file_offset` is a single contiguous run on disk)
/// and aligned to the stripe's offset + a multiple of [`UNIT_BYTES`].
#[derive(Clone, Copy, Debug)]
pub struct Unit {
    /// Absolute file offset in the source log.
    pub file_offset: u64,
    /// Offset within the chunk batch's pinned buffer.
    pub buf_offset: usize,
    /// Bytes to read.
    pub length: usize,
}

/// Per-record placement on the unit grid — `first_unit` is the unit
/// containing the record's first byte (where its header + metadata
/// live), `last_unit` is the unit containing its last byte. Equal
/// when the record fits in one unit; differ by 1 (or more, in
/// principle) when it spans a boundary.
#[derive(Clone, Copy, Debug)]
pub struct RecordUnitRange {
    pub first_unit: u32,
    pub last_unit: u32,
}

/// 1 MiB-granular unit plan for one [`ChunkBatch`]. Built by
/// [`plan_units`] in the [`pipeline`](super::pipeline) preparation
/// step; consumed by the three pipeline actors.
#[derive(Clone, Debug)]
pub struct UnitPlan {
    pub units: Vec<Unit>,
    /// Same length & order as `ChunkBatch.records`.
    pub record_units: Vec<RecordUnitRange>,
}

/// Pipeline unit size — matches the [`SUB_STRIPE_BYTES`] (1 MiB) used
/// by the direct-I/O read path so units = sub-stripes. A 64 MiB chunk
/// batch contains ≤ 64 units; a 1 MiB chunk batch is one unit.
///
/// [`SUB_STRIPE_BYTES`]: super::SUB_STRIPE_BYTES
pub const UNIT_BYTES: usize = 1024 * 1024;

/// Partition a [`ChunkBatch`] into 1 MiB units and map each record to
/// the unit range that holds its bytes. Stripes are partitioned
/// independently: a unit never crosses a stripe boundary, so each
/// unit reads a contiguous run from one file offset.
pub fn plan_units(batch: &ChunkBatch, unit_size: usize) -> UnitPlan {
    let mut units = Vec::new();
    for stripe in &batch.stripes {
        let mut off = 0usize;
        while off < stripe.len {
            let len = (stripe.len - off).min(unit_size);
            units.push(Unit {
                file_offset: stripe.file_offset + off as u64,
                buf_offset: stripe.buf_offset + off,
                length: len,
            });
            off += len;
        }
    }

    let record_units: Vec<RecordUnitRange> = batch
        .records
        .iter()
        .map(|rec| {
            let start = rec.buf_offset;
            // -1 because the record's last byte is at offset
            // `buf_offset + record_size - 1`; record_size > 0 per the
            // planner's records-have-bytes invariant.
            let last = rec.buf_offset + rec.record_size as usize - 1;
            RecordUnitRange {
                first_unit: find_unit(&units, start) as u32,
                last_unit: find_unit(&units, last) as u32,
            }
        })
        .collect();

    UnitPlan {
        units,
        record_units,
    }
}

fn find_unit(units: &[Unit], buf_offset: usize) -> usize {
    units
        .binary_search_by(|u| {
            if buf_offset < u.buf_offset {
                std::cmp::Ordering::Greater
            } else if buf_offset >= u.buf_offset + u.length {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .expect("buf_offset must fall within some unit (records-fit-in-stripes invariant)")
}

fn build_stripes_from_records(records: &[PlannedRecord]) -> Vec<ChunkStripe> {
    let mut out: Vec<ChunkStripe> = Vec::new();
    let mut it = records.iter();
    let Some(first) = it.next() else {
        return out;
    };
    let mut start_file = first.file_offset;
    let mut end_file = first.file_offset + first.record_size;
    let mut start_buf = first.buf_offset;
    for r in it {
        if r.file_offset == end_file {
            end_file = r.file_offset + r.record_size;
        } else {
            out.push(ChunkStripe {
                file_offset: start_file,
                len: (end_file - start_file) as usize,
                buf_offset: start_buf,
            });
            start_file = r.file_offset;
            end_file = r.file_offset + r.record_size;
            start_buf = r.buf_offset;
        }
    }
    out.push(ChunkStripe {
        file_offset: start_file,
        len: (end_file - start_file) as usize,
        buf_offset: start_buf,
    });
    out
}
